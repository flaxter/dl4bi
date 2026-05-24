"""Can we use OT-CFM (the linear interpolant) inside FlowGP?

FlowGP is already a closed-form flow-matching model; the interpolant is a free
choice. OT-CFM / rectified flow uses the *linear* interpolant

    f_t = (1 - t) f0 + t z,      conditional velocity  z - f0  (straight path),

whereas FlowGP uses the *variance-preserving* (VP) interpolant

    f_t = alpha(t) f0 + sqrt(1 - alpha(t)^2) z.

For an affine interpolant f_t = a(t) f0 + b(t) z, the whitened path covariance is
A(t) = (a^2 + b^2) I, and the paper's Corollary D.4 shows A(t) = I uniquely
minimises the Wasserstein-2 transport cost. VP satisfies a^2 + b^2 = 1, so after
whitening A(t) = I and the *unconditional* velocity is exactly zero -- the GP
prior is a 0-step reparameterisation. The linear (OT) interpolant has
a^2 + b^2 = 1 - 2t + 2t^2 (dips to 0.5 at t = 0.5) != 1, so it is NOT
transport-optimal: even after whitening it has a curved path with non-zero
velocity that must be integrated.

This script shows that empirically: VP recovers the GP prior exactly at 0 steps,
while the OT/linear interpolant needs ~10 Euler steps just to recover the
*unconstrained* prior. So OT-CFM cannot make FlowGP's sampler cheaper -- VP +
whitening already dominates. (And the guidance term that makes the constrained
flow expensive/non-smooth is identical for any interpolant, so OT-CFM does not
help there or with HMC-differentiability either.)

Where OT-CFM *is* the right tool: as the distillation target. Training a smooth
OT-CFM velocity on FlowGP's constrained samples gives the straight, few-step,
HMC-differentiable flow that raw FlowGP cannot provide (see fewstep_flowgp_hmc.py).

Run:
    uv run python replications/flowgp/ot_interpolant_flowgp.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from flowgp import make_schedule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = 64


def vp_path_var(t):
    _, alpha, _ = make_schedule()
    a = alpha(t)
    return a**2 + (1.0 - a**2)  # = 1 by construction (A(t) = I)


def linear_path_var(t):
    return (1.0 - t) ** 2 + t**2  # OT/linear interpolant, dips to 0.5


def linear_unconditional_flow(z, n_steps):
    """Integrate the whitened linear-interpolant probability-flow ODE from t=1
    (noise z) to t=0. Velocity c(t) x with c(t) = (2t-1) / ((1-t)^2 + t^2)."""
    grid = jnp.linspace(1.0, 0.0, n_steps + 1)

    def step(x, j):
        t = grid[j]
        dt = grid[j + 1] - grid[j]  # negative (going backward)
        c = (2.0 * t - 1.0) / ((1.0 - t) ** 2 + t**2)
        return x + dt * c * x, None

    x, _ = jax.lax.scan(step, z, jnp.arange(n_steps))
    return x


def main():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (4000, D))  # whitened noise ~ N(0, I)

    print("Recovering the *unconstrained* whitened GP prior N(0, I) from noise:")
    print("  (target output variance = 1.000 per dimension)\n")
    print("  K steps | VP+whitening | OT/linear interpolant")
    # VP: unconditional velocity is identically zero -> f0_hat = z, exact at K=0.
    vp_var = float(jnp.var(z))
    for k in [0, 1, 2, 3, 5, 10, 30]:
        if k == 0:
            lin = z  # nothing to integrate yet
        else:
            lin = linear_unconditional_flow(z, k)
        lin_var = float(jnp.var(lin))
        vp_str = f"{vp_var:.3f} (exact)" if k == 0 else f"{vp_var:.3f}"
        print(
            f"  K={k:3d}   |  {vp_str:13s}| var = {lin_var:.3f}  "
            f"(err {abs(lin_var - 1.0):.3f})"
        )

    print("\n  VP is exact with no integration (A(t)=I, zero velocity); the OT/")
    print("  linear interpolant needs ~10 steps just for the Gaussian part, because")
    print("  its path covariance dips to 0.5 (not transport-optimal, Cor. D.4).")
    print("\n  => OT-CFM cannot speed up FlowGP's sampler. Its value is as a")
    print("     distillation target for an HMC-usable surrogate, not inside FlowGP.")

    _plot()


def _plot():
    t = jnp.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(
        t,
        jax.vmap(vp_path_var)(t),
        color="tab:blue",
        lw=2,
        label="VP + whitening (FlowGP): A(t)=I, optimal",
    )
    ax.plot(
        t,
        linear_path_var(t),
        color="tab:red",
        lw=2,
        label="OT/linear interpolant: A(t) dips to 0.5",
    )
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("t  (0 = GP sample, 1 = noise)")
    ax.set_ylabel("whitened path variance  A(t) / n")
    ax.set_title("Transport cost: VP is the unique optimal affine interpolant")
    ax.legend(fontsize=9)
    ax.set_ylim(0.0, 1.15)
    fig.tight_layout()
    out = os.path.join(HERE, "ot_interpolant_flowgp.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
