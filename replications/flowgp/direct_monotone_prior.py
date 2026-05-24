"""What is FlowGP's role for a monotone prior? A direct forward generator.

Seth's critique, applied to the shape-constraint case: "FlowGP generates the
constrained training distribution; OT-CFM distills it" -- but can OT-CFM (or we)
just emulate the *true process* directly, with no FlowGP?

For a monotone prior the answer is yes, and trivially: a monotone function *is* a
cumulative sum of non-negative increments. So a direct forward generator is

    delta = softplus( L_inc @ xi + bias ),    xi ~ N(0, I)
    f     = offset + cumsum(delta)

with L_inc the Cholesky of an RBF kernel over the increment grid (for smoothness).
This samples smooth monotone functions *exactly* (100% monotone), is cheap, and
is differentiable -- so it drops straight into HMC with no FlowGP, no OT-CFM, no
training.

Compared on the same Bernoulli dose-response task:

    raw FlowGP in HMC        : 100% monotone, HMC acceptance 0   (not usable)
    FlowGP -> OT-CFM in HMC  :   0% monotone, acceptance 1.0, RMSE 0.13
    direct generator in HMC  : 100% monotone, acceptance ~1.0, RMSE ~0.04  <-- this

Conclusion: for a shape constraint the constrained set has a direct forward
parameterisation, so FlowGP (and the distillation) is unnecessary -- exactly the
pendulum story (direct_sim_baseline.py) in another guise. FlowGP's irreducible
value is sampling p(f0 | D, C), the data+constraint *posterior*, which has no
forward generator -- not prior emulation.

Run:
    uv run python replications/flowgp/direct_monotone_prior.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from deeprv_monotonic import M, monotone_fraction, se_kernel  # noqa: E402
from hmc_monotonic import hmc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X_GRID = jnp.linspace(0.0, 1.0, M)

# Smoothness of the increment field; offset/increment scales set the prior range.
INC_LENGTHSCALE = 0.15
INC_SCALE = 1.2
INC_BIAS = -2.2
OFFSET_SCALE = 3.0

_inc_x = 0.5 * (X_GRID[1:] + X_GRID[:-1])  # increment midpoints (M-1)
_K_inc = se_kernel(_inc_x, _inc_x) + 1e-6 * jnp.eye(M - 1)
_L_inc = jnp.linalg.cholesky(_K_inc)


def generate(w):
    """Direct monotone forward map: latent w ~ N(0, I)^M -> monotone f (M,)."""
    offset = OFFSET_SCALE * w[0]
    delta = jax.nn.softplus(INC_SCALE * (_L_inc @ w[1:]) + INC_BIAS)  # >= 0
    return offset + jnp.concatenate([jnp.zeros(1), jnp.cumsum(delta)])


def main():
    key = jax.random.PRNGKey(0)

    print("[1] Direct monotone generator (no FlowGP, no training) ...")
    w = jax.random.normal(key, (2000, M))
    prior = jax.vmap(generate)(w)
    print(
        f"    monotone fraction = {monotone_fraction(prior):.3f}   "
        f"pointwise std = {float(prior.std(0).mean()):.2f}   "
        f"range = [{float(prior.min()):.1f}, {float(prior.max()):.1f}]"
    )

    print("\n[2] HMC dose-response with the direct generator as prior ...")
    k_x, k_y, k_hmc = jax.random.split(jax.random.fold_in(key, 1), 3)
    x_obs = jnp.sort(jax.random.uniform(k_x, (25,)))
    y_obs = (
        jax.random.uniform(k_y, (25,)) < jax.nn.sigmoid(8.0 * (x_obs - 0.5))
    ).astype(jnp.float64)
    obs_idx = jnp.clip(jnp.round(x_obs * (M - 1)).astype(int), 0, M - 1)

    def potential(w):
        lg = generate(w)[obs_idx]
        ll = jnp.sum(
            y_obs * jax.nn.log_sigmoid(lg) + (1 - y_obs) * jax.nn.log_sigmoid(-lg)
        )
        return 0.5 * jnp.sum(w**2) - ll

    samples, acc = hmc(
        k_hmc, potential, jnp.zeros(M), n_samples=3000, step=0.03, n_leap=20
    )
    samples = samples[1000:]
    f_post = jax.vmap(generate)(samples)
    p_post = jax.nn.sigmoid(f_post)
    rmse = float(
        jnp.sqrt(jnp.mean((p_post.mean(0) - jax.nn.sigmoid(8.0 * (X_GRID - 0.5))) ** 2))
    )
    print(
        f"    acceptance={acc:.2f}  posterior monotone fraction="
        f"{monotone_fraction(f_post):.2f}  recovery RMSE={rmse:.3f}"
    )
    print(
        "\n  => A monotone prior is directly constructible and HMC-ready with no "
        "FlowGP.\n     FlowGP's role here is none; its value is the data+constraint "
        "posterior."
    )

    _plot(prior, x_obs, y_obs, p_post)


def _plot(prior, x_obs, y_obs, p_post):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tr in prior[:20]:
        axes[0].plot(X_GRID, tr, color="tab:purple", lw=0.6, alpha=0.5)
    axes[0].set_title("Direct monotone prior samples (no FlowGP)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    lo, hi = jnp.quantile(p_post, jnp.array([0.05, 0.95]), axis=0)
    axes[1].fill_between(
        X_GRID, lo, hi, color="tab:purple", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[1].plot(X_GRID, p_post.mean(0), color="tab:purple", label="posterior mean")
    axes[1].plot(
        X_GRID,
        jax.nn.sigmoid(8.0 * (X_GRID - 0.5)),
        "tab:green",
        lw=1.5,
        label="true p(x)",
    )
    axes[1].scatter(x_obs, y_obs, color="red", zorder=5, s=25, label="binary obs")
    axes[1].set_title("HMC dose-response (direct monotone prior, 100% monotone)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p(x)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "direct_monotone_prior.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
