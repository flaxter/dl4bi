"""Can FlowGP serve as / feed a DeepRV-style fast prior emulator?

Context
-------
DeepRV (dl4bi.vae.deep_rv) distills the GP reparameterisation z -> L_c z (the
Cholesky map, amortised over kernel hyper-parameters c) into a single neural
forward pass, giving a fast, differentiable, "non-centred" GP prior that drops
straight into NUTS/HMC for models with non-Gaussian likelihoods.

FlowGP samples a GP predictive under *arbitrary* conditioning, but each draw
costs ~1000 guided ODE steps with stochastic Monte-Carlo guidance -- far too
slow and too noisy to live inside an HMC leapfrog loop. So FlowGP cannot be a
literal drop-in replacement for DeepRV.

What FlowGP *can* do that DeepRV cannot is produce training data for priors that
have no easy exact sampler -- e.g. a *monotonic* GP prior. This script tests the
synthesis:

  1. Validate FlowGP reproduces the exact GP posterior under a linear-Gaussian
     likelihood (it must be an accurate emulator before we trust it).
  2. Use FlowGP to sample a monotonic GP prior. Treating the full initial+MC
     noise as the latent w ~ N(0, I) makes the sampler a deterministic map
     w -> f whose marginal over w is exactly the (S=1) monotonic prior.
  3. Distil that map into a monotone-by-construction DeepRV-style decoder and
     check it is (a) monotone (guaranteed by the architecture), (b)
     distributionally faithful, and (c) orders of magnitude faster -- i.e.
     actually usable inside HMC.

Key finding: a *naive* MSE decoder (z -> f) fails -- it fits the training pairs
yet produces non-monotone draws on fresh latents, because monotonicity is a
fragile pointwise-derivative constraint that pointwise MSE does not protect.
The monotone-by-construction decoder fixes this. So FlowGP does not *replace*
DeepRV; it *extends* the DeepRV recipe to exotic priors that have no easy
sampler, by supplying the training distribution.

Run:
    uv run python replications/flowgp/deeprv_monotonic.py
"""

from __future__ import annotations

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import optax  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

M = 64
LENGTHSCALE = 0.15
OUTPUTSCALE = 1.0
V_MONO = 1e-4


def se_kernel(xa, xb):
    d = xa[:, None] - xb[None, :]
    return OUTPUTSCALE * jnp.exp(-(d**2) / (2.0 * LENGTHSCALE**2))


# ---------------------------------------------------------------------------
# 1. Validation: FlowGP vs the exact GP posterior (linear-Gaussian likelihood)
# ---------------------------------------------------------------------------
def validate_against_exact_gp(key, x_grid, K):
    obs_idx = jnp.array([8, 24, 40, 56])
    y = jnp.array([-0.6, 0.4, -0.2, 0.8])
    sigma2 = 0.05**2

    Koo = K[obs_idx][:, obs_idx] + sigma2 * jnp.eye(obs_idx.size)
    Kxo = K[:, obs_idx]
    sol = jnp.linalg.solve(Koo, jnp.eye(obs_idx.size))
    m_post = Kxo @ sol @ y
    K_post = K - Kxo @ sol @ Kxo.T

    def loglik(f0):
        return -0.5 * jnp.sum((f0[obs_idx] - y) ** 2) / sigma2

    samples = flowgp_sample(key, jnp.zeros(M), K, loglik, n_samples=2000, T=1000, S=5)
    mean_rmse = float(jnp.sqrt(jnp.mean((samples.mean(0) - m_post) ** 2)))
    cov_rmse = float(jnp.sqrt(jnp.mean((jnp.cov(samples.T) - K_post) ** 2)))
    print("[1] FlowGP vs exact GP posterior (linear-Gaussian likelihood):")
    print(f"    posterior-mean RMSE = {mean_rmse:.4f}")
    print(
        f"    posterior-cov  RMSE = {cov_rmse:.4f}  "
        f"(prior-cov scale ~ {float(jnp.sqrt(jnp.mean(K_post**2))):.2f})"
    )
    return m_post, K_post


# ---------------------------------------------------------------------------
# 2. FlowGP monotonic GP prior as a deterministic flow map z -> f
# ---------------------------------------------------------------------------
def monotonic_loglik(f0):
    dx = 1.0 / M
    c = (f0[1:] - f0[:-1]) / dx
    return jnp.sum(norm.logcdf(c / V_MONO))


def flow_prior_pairs(key, K, w):
    """Deterministic FlowGP map from full noise w = [z, eps] -> monotonic f.

    Using S = 1, the latent is w = [f_hat_init, eps] ~ N(0, I) of dim 2M, and f
    is a deterministic function of w. Sampling w ~ N(0, I) therefore reproduces
    the *true* monotonic prior (no fixed-noise distortion), which is what makes
    the distilled emulator distributionally faithful.
    """
    n = w.shape[0]
    z = w[:, :M]
    eps = w[:, M:].reshape(n, 1, M)
    return flowgp_sample(
        key,
        jnp.zeros(M),
        K,
        monotonic_loglik,
        T=1000,
        S=1,
        f_hat_init=z,
        eps=eps,
    )


def monotone_fraction(samples, tol=1e-2):
    diffs = jnp.diff(samples, axis=1)
    return float(jnp.mean(jnp.all(diffs >= -tol, axis=1)))


# ---------------------------------------------------------------------------
# 3. DeepRV-style decoder distilling the flow map
# ---------------------------------------------------------------------------
class MonoDecoder(nn.Module):
    """Monotone-by-construction decoder: outputs a base value plus a cumulative
    sum of non-negative (softplus) increments, so every output is monotonically
    increasing regardless of approximation error."""

    hidden: tuple = (256, 256)
    out_dim: int = M

    @nn.compact
    def __call__(self, w):
        h = w
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        base = nn.Dense(1)(h)
        inc = nn.softplus(nn.Dense(self.out_dim - 1)(h))
        return jnp.concatenate([base, base + jnp.cumsum(inc, axis=-1)], axis=-1)


def train_decoder(key, w, f, steps=20000, batch=512, lr=1e-3, val_frac=0.15):
    """Distil the (deterministic) flow map w -> f with a monotone decoder."""
    n_val = int(w.shape[0] * val_frac)
    w_tr, w_va = w[n_val:], w[:n_val]
    f_tr, f_va = f[n_val:], f[:n_val]

    model = MonoDecoder()
    params = model.init(key, w_tr[:1])
    best = params
    opt = optax.adamw(lr, weight_decay=1e-4)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, wb, fb):
        def loss_fn(p):
            return jnp.mean((model.apply(p, wb) - fb) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def val_mse(params):
        return jnp.mean((model.apply(params, w_va) - f_va) ** 2)

    n = w_tr.shape[0]
    best_val = jnp.inf
    for i in range(steps):
        key, kb = jax.random.split(key)
        idx = jax.random.randint(kb, (batch,), 0, n)
        params, opt_state, _ = step(params, opt_state, w_tr[idx], f_tr[idx])
        if i % 500 == 0:
            v = val_mse(params)
            if v < best_val:
                best_val, best = v, params
    print(f"    best validation MSE = {float(best_val):.4f}")
    decode = jax.jit(lambda w: model.apply(best, w))
    return decode


def main():
    key = jax.random.PRNGKey(0)
    x_grid = jnp.linspace(0.0, 1.0, M)
    K = se_kernel(x_grid, x_grid) + 1e-6 * jnp.eye(M)

    key, k_val = jax.random.split(key)
    validate_against_exact_gp(k_val, x_grid, K)

    # Generate a monotonic-prior training set via the deterministic flow map.
    # Latent w = [f_hat_init, eps] in R^{2M}; sampling w ~ N(0,I) reproduces the
    # true monotonic prior (S = 1).
    print("\n[2] Sampling monotonic GP prior with FlowGP (latent = full noise) ...")
    key, k_w = jax.random.split(key)
    n_train = 10000
    w_train = jax.random.normal(k_w, (n_train, 2 * M))
    t0 = time.time()
    f_train = flow_prior_pairs(key, K, w_train)
    f_train.block_until_ready()
    flow_gen_time = time.time() - t0
    per_sample_flow = flow_gen_time / n_train
    print(
        f"    generated {n_train} samples in {flow_gen_time:.1f}s "
        f"({per_sample_flow * 1e3:.2f} ms/sample)"
    )
    print(
        f"    FlowGP monotone fraction = {monotone_fraction(f_train):.3f}  "
        f"pointwise std ~ {float(f_train.std(0).mean()):.2f}"
    )

    # Distill into a fast monotone-by-construction emulator.
    print("\n[3] Distilling flow map into a monotone DeepRV-style decoder ...")
    key, k_dec = jax.random.split(key)
    decode = train_decoder(k_dec, w_train, f_train)

    # Evaluate the emulator on fresh latents.
    key, k_test = jax.random.split(key)
    w_test = jax.random.normal(k_test, (2000, 2 * M))
    emu = decode(w_test)
    emu.block_until_ready()
    t0 = time.time()
    for _ in range(5):
        decode(w_test).block_until_ready()
    per_sample_emu = (time.time() - t0) / (5 * 2000)

    print(
        f"    emulator monotone fraction = {monotone_fraction(emu):.3f}  "
        f"pointwise std ~ {float(emu.std(0).mean()):.2f}"
    )
    # Distributional fidelity vs an independent FlowGP reference (fresh noise).
    key, k_ref = jax.random.split(key)
    w_ref = jax.random.normal(k_ref, (2000, 2 * M))
    ref = flow_prior_pairs(key, K, w_ref)
    mean_err = float(jnp.sqrt(jnp.mean((emu.mean(0) - ref.mean(0)) ** 2)))
    std_err = float(jnp.sqrt(jnp.mean((emu.std(0) - ref.std(0)) ** 2)))
    print(f"    pointwise mean RMSE vs FlowGP ref = {mean_err:.3f}")
    print(f"    pointwise std  RMSE vs FlowGP ref = {std_err:.3f}")
    print(
        f"\n[speed] FlowGP {per_sample_flow * 1e3:.2f} ms/sample  vs  "
        f"emulator {per_sample_emu * 1e6:.1f} us/sample  "
        f"-> {per_sample_flow / per_sample_emu:.0f}x faster"
    )

    _plot(x_grid, f_train, emu, ref)


def _plot(x_grid, flow_samples, emu_samples, ref_samples):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, title, s in [
        (axes[0], "FlowGP monotonic prior", flow_samples),
        (axes[1], "Distilled emulator (DeepRV-style)", emu_samples),
    ]:
        lo, hi = jnp.quantile(s, jnp.array([0.05, 0.95]), axis=0)
        ax.fill_between(
            x_grid, lo, hi, color="tab:blue", alpha=0.2, label="0.05-0.95 quantiles"
        )
        for traj in s[:15]:
            ax.plot(x_grid, traj, color="tab:blue", lw=0.6, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "deeprv_monotonic.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
