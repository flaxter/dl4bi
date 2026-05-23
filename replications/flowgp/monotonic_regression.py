"""Monotone and bounded GP regression with FlowGP (paper Appendix G / Figure 4).

Reproduces the shape-constrained regression sanity check from "Conditioning
Gaussian Processes on Almost Anything" (arXiv:2605.21041): a 1D GP conditioned
on seven (effectively noiseless) observations, additionally constrained to be
monotonically increasing and to lie between a known lower/upper envelope.

Run:
    uv run python replications/flowgp/monotonic_regression.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)  # GP conditioning needs float64

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Problem definition (Appendix G)
# ----------------------------------------------------------------------------
M = 64  # target grid size
LENGTHSCALE = 0.1  # kernel length-scale kappa (fixed/known)
OUTPUTSCALE = 0.25  # kernel variance tau^2 (fixed/known)
OBS_NOISE_VAR = 1e-10  # effectively noiseless observations
V_MONO = 1e-4  # probit sharpness for monotonicity
V_BOUND = 1e-5  # probit sharpness for boundedness


def f_true(x):
    """Ground-truth function (Eq. 61)."""
    return (jnp.arctan(20.0 * x - 10.0) - jnp.arctan(-10.0)) / 3.0


def lower_envelope(x):
    return jnp.zeros_like(x)


def upper_envelope(x):
    """u(x) = (1/3) log(30x + 1) + 0.1 (Eq. 64)."""
    return jnp.log(30.0 * x + 1.0) / 3.0 + 0.1


def se_kernel(xa, xb, lengthscale=LENGTHSCALE, outputscale=OUTPUTSCALE):
    """Squared-exponential covariance k(x, x') = tau^2 exp(-(x-x')^2 / 2 kappa^2)."""
    d = xa[:, None] - xb[None, :]
    return outputscale * jnp.exp(-(d**2) / (2.0 * lengthscale**2))


def gp_predictive(x_grid, x_obs, y_obs, noise_var=OBS_NOISE_VAR):
    """Closed-form zero-mean GP predictive N(m_pred, K_pred) on x_grid (Eqs. 2-3)."""
    K_oo = se_kernel(x_obs, x_obs) + noise_var * jnp.eye(x_obs.shape[0])
    K_go = se_kernel(x_grid, x_obs)
    K_gg = se_kernel(x_grid, x_grid)
    chol = jax.scipy.linalg.cho_factor(K_oo)
    m_pred = K_go @ jax.scipy.linalg.cho_solve(chol, y_obs)
    K_pred = K_gg - K_go @ jax.scipy.linalg.cho_solve(chol, K_go.T)
    K_pred = 0.5 * (K_pred + K_pred.T)  # symmetrise
    return m_pred, K_pred


def make_loglik(x_grid):
    """Build log p(C | f0) for the monotonicity (C1) and boundedness (C2) constraints.

    Both constraints are smoothed inequality conditions handled by a probit
    relaxation, log Phi(margin / v), recovering the hard constraint as v -> 0
    (Section 5.2, Eqs. 63-64).
    """
    dx = 1.0 / M
    u = upper_envelope(x_grid)
    ell = lower_envelope(x_grid)

    def loglik(f0):
        # C1: monotonicity via non-negative forward finite differences.
        c = (f0[1:] - f0[:-1]) / dx
        log_c1 = jnp.sum(norm.logcdf(c / V_MONO))
        # C2: boundedness via upper and lower margins.
        log_c2 = jnp.sum(norm.logcdf((u - f0) / V_BOUND)) + jnp.sum(
            norm.logcdf((f0 - ell) / V_BOUND)
        )
        return log_c1 + log_c2

    return loglik


def constraint_diagnostics(samples, x_grid, tol=1e-2):
    """Constraint satisfaction summary.

    These are *soft* (probit-relaxed) constraints, so we report both the
    fraction of samples satisfying them within a tolerance and the worst-case
    violation magnitude across all 100 samples. Residual violations are expected
    to be on the order of the probit bandwidths v (1e-4 to 1e-5).
    """
    diffs = jnp.diff(samples, axis=1)
    mono_viol = jnp.maximum(-diffs, 0.0)  # |negative| forward differences
    monotone = jnp.mean(jnp.all(diffs >= -tol, axis=1))

    u = upper_envelope(x_grid)
    ell = lower_envelope(x_grid)
    bound_viol = jnp.maximum(samples - u, 0.0) + jnp.maximum(ell - samples, 0.0)
    in_bounds = jnp.mean(jnp.all((samples <= u + tol) & (samples >= ell - tol), axis=1))
    return {
        "monotone_frac": float(monotone),
        "max_mono_violation": float(mono_viol.max()),
        "in_bounds_frac": float(in_bounds),
        "max_bound_violation": float(bound_viol.max()),
    }


def main():
    key = jax.random.PRNGKey(0)

    x_grid = jnp.linspace(0.0, 1.0, M)
    idx = jnp.arange(1, 8)  # i = 1, ..., 7
    x_obs = 0.1 + 1.0 / (idx + 1.0)
    y_obs = f_true(x_obs)

    m_pred, K_pred = gp_predictive(x_grid, x_obs, y_obs)
    loglik = make_loglik(x_grid)

    # Unconstrained GP samples (for the side-by-side comparison, Figure 4a).
    key, k_gp = jax.random.split(key)
    L = jnp.linalg.cholesky(K_pred + 1e-6 * jnp.eye(M))
    gp_samples = m_pred + jax.random.normal(k_gp, (100, M)) @ L.T

    # FlowGP samples conditioned on monotonicity + boundedness (Figure 4d).
    key, k_flow = jax.random.split(key)
    flow_samples = flowgp_sample(
        k_flow, m_pred, K_pred, loglik, n_samples=100, T=1000, S=5
    )

    gp_diag = constraint_diagnostics(gp_samples, x_grid)
    flow_diag = constraint_diagnostics(flow_samples, x_grid)
    print("Constraint satisfaction over 100 samples (tol=1e-2):")
    for name, d in [("unconstrained GP", gp_diag), ("FlowGP", flow_diag)]:
        print(
            f"  {name:18s}: monotone={d['monotone_frac']:.2f} "
            f"(max viol {d['max_mono_violation']:.1e})  "
            f"in-bounds={d['in_bounds_frac']:.2f} "
            f"(max viol {d['max_bound_violation']:.1e})"
        )

    _plot(x_grid, x_obs, y_obs, gp_samples, flow_samples)


def _panel(ax, title, x_grid, x_obs, y_obs, samples):
    lo, hi = jnp.quantile(samples, jnp.array([0.05, 0.95]), axis=0)
    ax.fill_between(
        x_grid, lo, hi, color="tab:blue", alpha=0.2, label="0.05-0.95 quantiles"
    )
    for s in samples[:10]:
        ax.plot(x_grid, s, color="tab:blue", lw=0.7, alpha=0.6)
    ax.plot(x_grid, upper_envelope(x_grid), "k--", lw=1.0, label="bounds")
    ax.plot(x_grid, lower_envelope(x_grid), "k--", lw=1.0)
    ax.plot(x_grid, f_true(x_grid), color="tab:green", lw=1.5, label="ground truth")
    ax.scatter(x_obs, y_obs, color="red", zorder=5, s=30, label="observations")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylim(-0.6, 1.2)


def _plot(x_grid, x_obs, y_obs, gp_samples, flow_samples):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _panel(axes[0], "(a) Unconstrained GP", x_grid, x_obs, y_obs, gp_samples)
    _panel(axes[1], "(d) FlowGP (ours)", x_grid, x_obs, y_obs, flow_samples)
    axes[0].set_ylabel("f(x)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "monotonic_regression.png")
    fig.savefig(out, dpi=150)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
