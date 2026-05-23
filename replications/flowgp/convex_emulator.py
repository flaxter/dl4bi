"""A harder constraint than monotonicity: a *convex* GP prior emulator.

Follows the same FlowGP -> DeepRV-style distillation recipe as
``deeprv_monotonic.py`` but for a second-order shape constraint -- convexity
(f'' >= 0). Convexity is a strictly harder target: it constrains curvature
rather than slope, so it is far more fragile under approximation error (a tiny
pointwise error easily flips the sign of a second difference). This makes the
"architecture must respect the constraint" lesson even sharper.

We compare two distilled emulators on FlowGP's convex prior:
  * a naive MLP decoder (z -> f), and
  * a convex-by-construction decoder (f built from a free base/slope plus a
    cumulative sum of non-negative second differences).

Then we use the convex emulator as a differentiable prior inside HMC for convex
regression. Constraints handled by the *architecture* (monotone, convex) are the
easy wins; FlowGP's real role is supplying the training distribution.

Run:
    uv run python replications/flowgp/convex_emulator.py
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
from hmc_monotonic import hmc  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

M = 64
LENGTHSCALE = 0.15
V_CONVEX = 1e-4  # probit sharpness on the (scaled) second difference


def se_kernel(xa, xb):
    d = xa[:, None] - xb[None, :]
    return jnp.exp(-(d**2) / (2.0 * LENGTHSCALE**2))


def convex_loglik(f0):
    """log p(C | f0): probit relaxation of the convexity constraint f'' >= 0."""
    dx = 1.0 / M
    d2 = (f0[2:] - 2.0 * f0[1:-1] + f0[:-2]) / dx**2
    return jnp.sum(norm.logcdf(d2 / (V_CONVEX / dx**2)))


def flow_prior_pairs(key, K, w):
    """Deterministic FlowGP map from full noise w = [z, eps] -> convex f (S=1)."""
    n = w.shape[0]
    z = w[:, :M]
    eps = w[:, M:].reshape(n, 1, M)
    return flowgp_sample(
        key, jnp.zeros(M), K, convex_loglik, T=1000, S=1, f_hat_init=z, eps=eps
    )


def convex_fraction(samples, tol=1e-2):
    d2 = jnp.diff(samples, n=2, axis=1)
    return float(jnp.mean(jnp.all(d2 >= -tol, axis=1)))


# ---------------------------------------------------------------------------
# Two decoders: a naive MLP and a convex-by-construction one.
# ---------------------------------------------------------------------------
class NaiveDecoder(nn.Module):
    hidden: tuple = (256, 256)
    out_dim: int = M

    @nn.compact
    def __call__(self, w):
        h = w
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        return nn.Dense(self.out_dim)(h)


class ConvexDecoder(nn.Module):
    """f built from a free base value and first slope plus a cumulative sum of
    non-negative (softplus) second differences -> convex by construction."""

    hidden: tuple = (256, 256)
    out_dim: int = M

    @nn.compact
    def __call__(self, w):
        h = w
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        base = nn.Dense(1)(h)
        slope0 = nn.Dense(1)(h)
        curv = nn.softplus(nn.Dense(self.out_dim - 2)(h))  # second diffs >= 0
        d = jnp.concatenate([slope0, slope0 + jnp.cumsum(curv, axis=-1)], axis=-1)
        return jnp.concatenate([base, base + jnp.cumsum(d, axis=-1)], axis=-1)


def train_decoder(key, model, w, f, steps=20000, batch=512, lr=1e-3, val_frac=0.15):
    n_val = int(w.shape[0] * val_frac)
    w_tr, w_va, f_tr, f_va = w[n_val:], w[:n_val], f[n_val:], f[:n_val]
    params = best = model.init(key, w_tr[:1])
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
    return jax.jit(lambda w: model.apply(best, w)), float(best_val)


def main():
    key = jax.random.PRNGKey(0)
    x_grid = jnp.linspace(0.0, 1.0, M)
    K = se_kernel(x_grid, x_grid) + 1e-6 * jnp.eye(M)

    print("[1] Sampling a convex GP prior with FlowGP (latent = full noise) ...")
    key, k_w = jax.random.split(key)
    n_train = 10000
    w_train = jax.random.normal(k_w, (n_train, 2 * M))
    t0 = time.time()
    f_train = flow_prior_pairs(key, K, w_train)
    f_train.block_until_ready()
    per_sample_flow = (time.time() - t0) / n_train
    print(
        f"    {n_train} samples in {per_sample_flow * n_train:.1f}s "
        f"({per_sample_flow * 1e3:.2f} ms/sample); "
        f"convex fraction = {convex_fraction(f_train):.3f}"
    )

    print("\n[2] Distilling: naive MLP vs convex-by-construction decoder ...")
    key, k1, k2 = jax.random.split(key, 3)
    naive_decode, naive_val = train_decoder(
        k1, NaiveDecoder(), w_train, f_train, steps=10000
    )
    convex_decode, convex_val = train_decoder(k2, ConvexDecoder(), w_train, f_train)

    key, k_test = jax.random.split(key)
    w_test = jax.random.normal(k_test, (2000, 2 * M))
    naive = naive_decode(w_test)
    convex = convex_decode(w_test)
    naive.block_until_ready()
    convex.block_until_ready()
    t0 = time.time()
    for _ in range(5):
        convex_decode(w_test).block_until_ready()
    per_sample_emu = (time.time() - t0) / (5 * 2000)

    key, k_ref = jax.random.split(key)
    ref = flow_prior_pairs(key, K, jax.random.normal(k_ref, (2000, 2 * M)))
    for name, val, s in [
        ("naive MLP", naive_val, naive),
        ("convex-by-constr.", convex_val, convex),
    ]:
        mean_err = float(jnp.sqrt(jnp.mean((s.mean(0) - ref.mean(0)) ** 2)))
        std_err = float(jnp.sqrt(jnp.mean((s.std(0) - ref.std(0)) ** 2)))
        print(
            f"    {name:18s}: val MSE={val:.3f}  convex frac={convex_fraction(s):.3f}  "
            f"mean RMSE={mean_err:.3f}  std RMSE={std_err:.3f}"
        )
    print(
        f"\n[speed] FlowGP {per_sample_flow * 1e3:.2f} ms/sample  vs  "
        f"emulator {per_sample_emu * 1e6:.1f} us/sample  "
        f"-> {per_sample_flow / per_sample_emu:.0f}x faster"
    )

    # [3] Drop-in HMC: convex regression with the convex emulator as the prior.
    print("\n[3] HMC convex regression with the convex emulator as prior ...")
    key, k_x, k_y, k_hmc = jax.random.split(key, 4)
    n_obs = 15

    def true_f(x):
        return 6.0 * (x - 0.4) ** 2 - 0.5

    x_obs = jnp.sort(jax.random.uniform(k_x, (n_obs,)))
    obs_idx = jnp.clip(jnp.round(x_obs * (M - 1)).astype(int), 0, M - 1)
    sigma = 0.15
    y_obs = true_f(x_obs) + sigma * jax.random.normal(k_y, (n_obs,))

    def decode1(w):
        return convex_decode(w[None, :])[0]

    def potential(w):
        f = decode1(w)
        ll = -0.5 * jnp.sum((f[obs_idx] - y_obs) ** 2) / sigma**2
        return 0.5 * jnp.sum(w**2) - ll

    # MAP init: the double-cumsum decoder makes the potential stiff, so we first
    # descend to a good starting point, then run HMC with a small step.
    w0 = jnp.zeros(2 * M)
    map_opt = optax.adam(5e-3)
    map_state = map_opt.init(w0)

    @jax.jit
    def map_step(w, state):
        g = jax.grad(potential)(w)
        updates, state = map_opt.update(g, state)
        return optax.apply_updates(w, updates), state

    for _ in range(3000):
        w0, map_state = map_step(w0, map_state)

    samples, acc = hmc(k_hmc, potential, w0, n_samples=3000, step=0.004, n_leap=20)
    samples = samples[1000:]
    f_post = jax.vmap(decode1)(samples)
    rmse = float(jnp.sqrt(jnp.mean((f_post.mean(0) - true_f(x_grid)) ** 2)))
    print(
        f"    acceptance={acc:.2f}  posterior convex frac={convex_fraction(f_post):.3f}"
        f"  posterior-mean RMSE vs truth={rmse:.3f}"
    )

    _plot(x_grid, f_train, naive, convex, x_obs, y_obs, f_post, true_f)


def _plot(x_grid, flow, naive, convex, x_obs, y_obs, f_post, true_f):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, title, s in [
        (axes[0], "FlowGP convex prior", flow),
        (axes[1], "Naive MLP emulator", naive),
    ]:
        lo, hi = jnp.quantile(s, jnp.array([0.05, 0.95]), axis=0)
        ax.fill_between(x_grid, lo, hi, color="tab:blue", alpha=0.2)
        for traj in s[:15]:
            ax.plot(x_grid, traj, color="tab:blue", lw=0.6, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
    # convex-by-construction HMC posterior on the third panel
    lo, hi = jnp.quantile(f_post, jnp.array([0.05, 0.95]), axis=0)
    axes[2].fill_between(
        x_grid, lo, hi, color="tab:blue", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[2].plot(x_grid, f_post.mean(0), color="tab:blue", label="posterior mean")
    axes[2].plot(x_grid, true_f(x_grid), "tab:green", lw=1.5, label="convex truth")
    axes[2].scatter(x_obs, y_obs, color="red", zorder=5, s=25, label="noisy obs")
    axes[2].set_title("Convex regression via HMC (convex emulator prior)")
    axes[2].set_xlabel("x")
    axes[2].legend(fontsize=8)
    axes[0].set_ylabel("f(x)")
    fig.tight_layout()
    out = os.path.join(HERE, "convex_emulator.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
