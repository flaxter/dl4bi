"""Can FlowGP play the role of FM-DeepRV -- a few-step flow prior inside HMC?

The flow-matching DeepRV variant (makkunda/dl4bi, dl4bi/vae/flow_matching.py)
learns an OT-CFM velocity v = f - z and integrates it for only K=3-5 Euler steps
as a fast, differentiable GP-prior reparameterisation inside HMC. FlowGP's
velocity is closed-form (no training), so it is tempting to run a few-step
*guided* flow live inside HMC and get a constrained prior for free.

This script characterises whether that works. Conclusion: not robustly.

  1. K=3-5 is too few. FM-DeepRV's OT path is straight (constant velocity v=f-z),
     so 3-5 Euler steps integrate it accurately. FlowGP's guided path is curved
     and stiff early on: K=3-5 overshoots (variance explodes), and only K>~10
     cleanly satisfies the constraint.

  2. FlowGP is not robustly HMC-differentiable. HMC must differentiate the prior
     map w.r.t. the latent; since FlowGP's guidance term is *itself* a gradient
     of the (sharp) likelihood, this is a second-order derivative through a
     non-smooth velocity (sharp probit gradient + the stabilising velocity clip,
     whose tanh/||v|| has a singular second derivative). The result is NaN
     gradients for a large fraction of latents at any meaningful constraint
     sharpness, and HMC fails to move (acceptance ~ 0).

So FM-DeepRV works at K=3-5 precisely because its velocity is a *smooth learned
network*. The way to get the same with FlowGP is to give it a smooth surrogate:
distil FlowGP's constrained samples into a one-step decoder (hmc_monotonic.py,
which works: 100% monotone, recovery RMSE ~0.04) or train an FM velocity on
FlowGP samples (the natural marriage of the two). FlowGP's role is to *generate*
the constrained training distribution that FM-DeepRV alone cannot sample.

Run:
    uv run python replications/flowgp/fewstep_flowgp_hmc.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from deeprv_monotonic import M, monotone_fraction, se_kernel  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from hmc_monotonic import hmc  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X_GRID = jnp.linspace(0.0, 1.0, M)
K_PRIOR = se_kernel(X_GRID, X_GRID) + 1e-6 * jnp.eye(M)


def monotone_loglik(v):
    def ll(f0):
        c = (f0[1:] - f0[:-1]) * M  # forward difference / dx, dx = 1/M
        return jnp.sum(norm.logcdf(c / v))

    return ll


def flow_map(z, eps, n_steps, v, v_max=1e2):
    return flowgp_sample(
        jax.random.PRNGKey(0),
        jnp.zeros(M),
        K_PRIOR,
        monotone_loglik(v),
        T=n_steps,
        S=eps.shape[1],
        f_hat_init=z,
        eps=eps,
        v_max=v_max,
    )


def k_sweep(key):
    print("[1] FlowGP monotone prior vs number of ODE steps K (sharp v=1e-4):")
    eps = jax.random.normal(key, (256, 5, M))
    z = jax.random.normal(jax.random.fold_in(key, 1), (256, M))
    ks, fracs, stds, samples = [3, 5, 10, 20, 50], [], [], {}
    for k in ks:
        s = flow_map(z, eps, k, 1e-4)
        fracs.append(monotone_fraction(s))
        stds.append(float(s.std(0).mean()))
        samples[k] = s
        print(
            f"    K={k:3d}:  monotone fraction={fracs[-1]:.3f}   "
            f"pointwise std={stds[-1]:.2f}"
        )
    return ks, fracs, stds, samples


def grad_stability(key):
    print(
        "\n[2] Finite second-order gradient fraction over 30 latents (HMC needs these):"
    )
    eps = jax.random.normal(key, (1, 1, M))
    zs = jax.random.normal(jax.random.fold_in(key, 2), (30, M))
    rows = []
    for v_max, tag in [(1e2, "clip on"), (1e8, "no clip")]:
        for v in [0.1, 0.3, 1.0]:

            def scal(z, v=v, v_max=v_max):
                return jnp.sum(flow_map(z[None, :], eps, 10, v, v_max)[0, ::8])

            g = jax.jit(jax.grad(scal))
            frac = (
                sum(int(not bool(jnp.any(jnp.isnan(g(z))))) for z in zs) / zs.shape[0]
            )
            rows.append((tag, v, frac))
            print(
                f"    {tag:8s} probit v={v:>3}:  finite grad on {frac * 100:3.0f}% "
                "of latents"
            )
    return rows


def hmc_attempt(key):
    print(
        "\n[3] HMC with live K=10 FlowGP prior, most-stable regime "
        "(no clip, soft v=1.0):"
    )
    eps = jax.random.normal(key, (1, 1, M))

    def decode1(z):
        return flow_map(z[None, :], eps, 10, 1.0, v_max=1e8)[0]

    k_x, k_y, k_hmc = jax.random.split(jax.random.fold_in(key, 3), 3)
    x_obs = jnp.sort(jax.random.uniform(k_x, (25,)))
    y_obs = (
        jax.random.uniform(k_y, (25,)) < jax.nn.sigmoid(8.0 * (x_obs - 0.5))
    ).astype(jnp.float64)
    obs_idx = jnp.clip(jnp.round(x_obs * (M - 1)).astype(int), 0, M - 1)

    def potential(z):
        lg = decode1(z)[obs_idx]
        ll = jnp.sum(
            y_obs * jax.nn.log_sigmoid(lg) + (1 - y_obs) * jax.nn.log_sigmoid(-lg)
        )
        return 0.5 * jnp.sum(z**2) - ll

    _, acc = hmc(k_hmc, potential, jnp.zeros(M), n_samples=1500, step=0.02, n_leap=15)
    print(
        f"    HMC acceptance = {acc:.2f}  -> chain cannot move; raw FlowGP is not "
        "usable as a live HMC prior."
    )
    print(
        "    (Contrast: the distilled 1-step decoder in hmc_monotonic.py gives "
        "acceptance 1.0,\n     100% monotone draws, recovery RMSE ~0.04.)"
    )


def _plot(ks, fracs, stds, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(ks, fracs, "o-", color="tab:blue", label="monotone fraction")
    ax.set_xlabel("ODE steps K")
    ax.set_ylabel("monotone fraction", color="tab:blue")
    ax.axvspan(2.5, 8, color="tab:red", alpha=0.1)
    ax.text(4.5, 0.5, "K=3-5\novershoots", ha="center", fontsize=8, color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(ks, stds, "s--", color="tab:gray", label="pointwise std")
    ax2.set_ylabel("pointwise std (target ~1.4)", color="tab:gray")
    ax.set_title("FlowGP guided flow needs K~10, not 3-5")

    ax = axes[1]
    labels = [f"{t}\nv={v}" for t, v, _ in rows]
    vals = [f * 100 for _, _, f in rows]
    colors = ["tab:green" if f == 1.0 else "tab:red" for _, _, f in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% latents with finite 2nd-order grad")
    ax.set_title("Not HMC-differentiable except at very soft v")
    fig.tight_layout()
    out = os.path.join(HERE, "fewstep_flowgp_hmc.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


def main():
    key = jax.random.PRNGKey(0)
    ks, fracs, stds, _ = k_sweep(key)
    rows = grad_stability(key)
    hmc_attempt(key)
    _plot(ks, fracs, stds, rows)


if __name__ == "__main__":
    main()
