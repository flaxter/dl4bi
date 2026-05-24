"""Benchmark: FlowGP guidance vs neural SBI (amortized NPE) on the SIR task.

Same problem as simulator_conditioning.py -- infer a GP-distributed log beta(t)
from a noisy incidence curve produced by a stochastic SIR -- compared two ways:

  * FlowGP gradient-free guidance: NON-amortized. No training, but pays
    T*S*N*R simulator runs *per* inference.
  * Neural posterior estimation (NPE): AMORTIZED. Sample (theta, x) pairs once
    from prior x simulator, train a conditional posterior q_phi(theta | x), then
    get a near-free posterior for ANY observed curve. Here q_phi is a conditional
    Gaussian in the GP-whitened coordinates (a robust, standard amortized
    baseline; a normalizing flow would be more expressive).

The honest axes: recovery quality, and -- crucially -- *simulator budget* and
amortization. NPE trades a fixed training budget for instant reuse; FlowGP pays
per inference and reuses nothing.

Run:
    uv run python replications/flowgp/npe_vs_flowgp.py
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
from simulator_conditioning import (  # noqa: E402
    PRIOR_MEAN,
    TGRID,
    M,
    infer_beta,
    se_kernel,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class NPEGaussian(nn.Module):
    """Amortized conditional posterior q(theta_white | x) = N(mu(x), diag)."""

    hidden: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        h = x
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        return nn.Dense(M)(h), nn.Dense(M)(h)  # mu, log-var


def feat(inc):
    return jnp.log1p(inc)  # compress the dynamic range of incidence


def train_npe(key, L, n_train=20000, steps=8000, batch=512):
    k_th, k_sim, k_init = jax.random.split(key, 3)
    theta = jax.random.normal(k_th, (n_train, M))  # whitened GP prior
    logbeta = theta @ L.T + PRIOR_MEAN
    x = feat(jax.vmap(simulate)(logbeta, jax.random.split(k_sim, n_train)))
    x_mean, x_std = x.mean(0), x.std(0) + 1e-6
    xn = (x - x_mean) / x_std

    model = NPEGaussian()
    params = model.init(k_init, xn[:1])
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, key):
        idx = jax.random.randint(key, (batch,), 0, n_train)

        def loss_fn(p):
            mu, lv = model.apply(p, xn[idx])
            return jnp.mean(0.5 * (lv + (theta[idx] - mu) ** 2 / jnp.exp(lv)))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for _ in range(steps):
        key, k = jax.random.split(key)
        params, opt_state, _ = step(params, opt_state, k)

    def posterior(key, inc_obs, n):
        mu, lv = model.apply(params, ((feat(inc_obs) - x_mean) / x_std)[None])
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, (n, M))
        return z @ L.T + PRIOR_MEAN  # log beta samples

    return posterior, n_train


def metrics(name, post_logbeta, true_beta, obs, key, sim_budget):
    beta = jnp.exp(post_logbeta)
    rmse = float(jnp.sqrt(jnp.mean((beta.mean(0) - true_beta) ** 2)))
    pp = jax.vmap(simulate)(post_logbeta, jax.random.split(key, beta.shape[0]))
    pp_rmse = float(jnp.sqrt(jnp.mean((pp.mean(0) - obs) ** 2)))
    print(
        f"  {name:8s}: beta RMSE={rmse:.3f}  predictive RMSE={pp_rmse:.1f}  "
        f"simulator runs={sim_budget:,}"
    )
    return beta, pp


def main():
    key = jax.random.PRNGKey(0)
    t = jnp.arange(M)
    true_beta = jnp.where(t < 20, 0.45, jnp.where(t < 40, 0.09, 0.18))
    K = se_kernel(TGRID, TGRID)
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(M))

    key, k_obs = jax.random.split(key)
    obs = simulate(jnp.log(true_beta), k_obs)
    print(f"observed: peak {float(obs.max()):.0f}, total {float(obs.sum()):.0f}\n")

    # --- neural SBI (amortized NPE) ---
    print("[NPE] training amortized conditional posterior ...")
    t0 = time.time()
    key, k_tr = jax.random.split(key)
    posterior, n_train = train_npe(k_tr, L)
    npe_train_time = time.time() - t0
    key, k_ps, k_m = jax.random.split(key, 3)
    npe_logbeta = posterior(k_ps, obs, 60)

    # --- FlowGP (non-amortized guidance) ---
    print("[FlowGP] gradient-free guidance (per-inference simulator runs) ...")
    t0 = time.time()
    key, k_fl = jax.random.split(key)
    flow_T, flow_S, flow_N, flow_R = 150, 32, 40, 3
    flow_logbeta = infer_beta(k_fl, K, obs, n_samples=flow_N, T=flow_T, S=flow_S)
    flow_time = time.time() - t0
    flow_budget = flow_T * flow_S * flow_N * flow_R

    print("\nResults (same observed curve):")
    key, k1, k2 = jax.random.split(key, 3)
    npe_beta, npe_pp = metrics("NPE", npe_logbeta, true_beta, obs, k1, n_train)
    flow_beta, flow_pp = metrics(
        "FlowGP", flow_logbeta, true_beta, obs, k2, flow_budget
    )
    print(
        f"\n  train/infer time: NPE {npe_train_time:.0f}s (amortized, reused free) "
        f"vs FlowGP {flow_time:.0f}s (per inference)"
    )
    print(
        f"  simulator budget: NPE {n_train:,} (one-off) "
        f"vs FlowGP {flow_budget:,} per inference "
        f"-> {flow_budget / n_train:.0f}x more, every time"
    )
    print(
        "  amortization: a new outbreak/region is ~free for NPE, full cost for FlowGP"
    )

    _plot(t, true_beta, npe_beta, flow_beta, obs, npe_pp, flow_pp)


def _plot(t, true_beta, npe_beta, flow_beta, obs, npe_pp, flow_pp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for beta, c, lbl in [
        (npe_beta, "tab:orange", "NPE"),
        (flow_beta, "tab:blue", "FlowGP"),
    ]:
        lo, hi = jnp.quantile(beta, jnp.array([0.05, 0.95]), axis=0)
        axes[0].fill_between(t, lo, hi, color=c, alpha=0.15)
        axes[0].plot(t, beta.mean(0), color=c, label=f"{lbl} posterior mean")
    axes[0].plot(t, true_beta, "tab:green", lw=2, label="true beta(t)")
    axes[0].set_title("beta(t) recovery: NPE vs FlowGP")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("beta(t)")
    axes[0].legend(fontsize=8)

    for pp, c, lbl in [(npe_pp, "tab:orange", "NPE"), (flow_pp, "tab:blue", "FlowGP")]:
        axes[1].plot(t, pp.mean(0), color=c, label=f"{lbl} predictive")
    axes[1].scatter(t, obs, color="red", s=12, zorder=5, label="observed")
    axes[1].set_title("Posterior-predictive fit")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("daily incidence")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "npe_vs_flowgp.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
