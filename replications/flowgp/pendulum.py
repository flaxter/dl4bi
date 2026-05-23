"""Physics-informed GP regression: the damped nonlinear pendulum (Appendix H.2).

Reproduces the pendulum panel of Figure 1(a) from "Conditioning Gaussian
Processes on Almost Anything" (arXiv:2605.21041). A GP is conditioned on sparse,
noisy observations of the angular trajectory theta(t), and FlowGP additionally
enforces the governing equation of motion

    theta''(t) + sin(theta(t)) + beta theta'(t) = 0,   beta = 0.2,

as a Gaussian likelihood over the point-wise ODE residual (central finite
differences on the grid). The unconstrained GP is coherent but reverts toward
the prior away from data; FlowGP recovers the physically-consistent decaying
oscillation across the whole horizon.

The paper uses the train/test split from PHYSS [31], which is not public, so we
generate data from our own high-accuracy numerical solution. Absolute RMSE/NLPD
therefore will not match the paper exactly, but the qualitative behaviour and
the GP-vs-FlowGP comparison reproduce.

Run:
    uv run python replications/flowgp/pendulum.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from gp_utils import (  # noqa: E402
    extend_to_test,
    fit_gp,
    gp_posterior,
    predictive_metrics,
)
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

T_HORIZON = 30.0  # seconds
BETA_DAMP = 0.2  # damping coefficient
THETA0 = 2.5  # initial angle (released from rest)
M = 200  # grid size
N_OBS = 18  # number of training observations
OBS_NOISE_STD = 0.05  # measurement-error standard deviation
# ODE-residual likelihood std. Because the whitened velocity is clipped to
# v_max, only the residual-gradient *direction* matters once the clip saturates,
# so results are identical for any SIGMA_PHYS in ~[1e-8, 1e-4]; the paper's
# literal 1e-10 just overflows float64 with no change in the clipped regime.
SIGMA_PHYS = 1e-6
N_SAMPLES = 100


def solve_pendulum(t_eval):
    """High-accuracy numerical solution of the damped nonlinear pendulum."""

    def rhs(_t, y):
        theta, omega = y
        return [omega, -np.sin(theta) - BETA_DAMP * omega]

    sol = solve_ivp(
        rhs,
        (0.0, T_HORIZON),
        [THETA0, 0.0],
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
        method="DOP853",
    )
    return sol.y[0]


def make_physics_loglik(grid_t):
    """log p(C | f0): Gaussian likelihood on the ODE residual at interior nodes.

    Derivatives are central finite differences in *real* time (the kernel works
    on normalised time, but the equation of motion is in seconds).
    """
    dt = float(grid_t[1] - grid_t[0])

    def loglik(f0):
        fp = (f0[2:] - f0[:-2]) / (2.0 * dt)  # theta'(t)
        fpp = (f0[2:] - 2.0 * f0[1:-1] + f0[:-2]) / dt**2  # theta''(t)
        r = fpp + jnp.sin(f0[1:-1]) + BETA_DAMP * fp  # residual (Eq. 71)
        return -0.5 * jnp.sum(r**2) / SIGMA_PHYS**2

    return loglik


def physics_residual_rms(samples, grid_t):
    """Root-mean-square ODE residual across a set of trajectories on the grid."""
    dt = float(grid_t[1] - grid_t[0])
    fp = (samples[:, 2:] - samples[:, :-2]) / (2.0 * dt)
    fpp = (samples[:, 2:] - 2.0 * samples[:, 1:-1] + samples[:, :-2]) / dt**2
    r = fpp + jnp.sin(samples[:, 1:-1]) + BETA_DAMP * fp
    return float(jnp.sqrt(jnp.mean(r**2)))


def main():
    rng = np.random.default_rng(0)

    # Grid in real time [0, 30]; the GP kernel uses normalised time t / 30.
    grid_t = jnp.linspace(0.0, T_HORIZON, M)
    Xg = (grid_t / T_HORIZON)[:, None]

    # Sparse, noisy training observations; denser held-out test set.
    t_obs = np.sort(rng.uniform(0.5, T_HORIZON, size=N_OBS))
    y_obs = solve_pendulum(t_obs) + rng.normal(0.0, OBS_NOISE_STD, size=N_OBS)
    t_test = np.linspace(0.2, T_HORIZON - 0.2, 120)
    y_test = solve_pendulum(t_test)

    Xo = jnp.asarray(t_obs / T_HORIZON)[:, None]
    yo = jnp.asarray(y_obs)
    Xt = jnp.asarray(t_test / T_HORIZON)[:, None]
    yt = jnp.asarray(y_test)

    # Fit GP hyperparameters + affine mean by marginal likelihood on D only.
    params = fit_gp(Xo, yo, noise_var=OBS_NOISE_STD**2)
    print(
        f"fitted: lengthscale={float(jnp.exp(params['log_ls'][0])):.3f} "
        f"outputscale={float(jnp.exp(params['log_os'])):.3f} "
        f"mean(a={float(params['a']):.2f}, b={float(params['b'][0]):.2f})  "
        f"nll={params['nll']:.2f}"
    )

    mean_fn, cov_fn = gp_posterior(Xo, yo, params)
    m_pred = mean_fn(Xg)
    K_pred = 0.5 * (cov_fn(Xg, Xg) + cov_fn(Xg, Xg).T)

    loglik = make_physics_loglik(grid_t)

    # Unconstrained GP samples on the grid.
    key = jax.random.PRNGKey(1)
    key, k_gp = jax.random.split(key)
    Lg = jnp.linalg.cholesky(K_pred + 1e-8 * jnp.eye(M))
    gp_samples = m_pred + jax.random.normal(k_gp, (N_SAMPLES, M)) @ Lg.T

    # FlowGP samples enforcing the ODE.
    key, k_flow = jax.random.split(key)
    flow_samples = flowgp_sample(
        k_flow, m_pred, K_pred, loglik, n_samples=N_SAMPLES, T=1000, S=5
    )

    # Off-grid test predictions (Eq. 65) and metrics.
    gp_test = extend_to_test(gp_samples, Xg, Xt, m_pred, K_pred, mean_fn, cov_fn)
    flow_test = extend_to_test(flow_samples, Xg, Xt, m_pred, K_pred, mean_fn, cov_fn)
    gp_rmse, gp_nlpd = predictive_metrics(gp_test, yt, OBS_NOISE_STD**2)
    flow_rmse, flow_nlpd = predictive_metrics(flow_test, yt, OBS_NOISE_STD**2)

    print(
        f"ODE residual RMS (grid): GP={physics_residual_rms(gp_samples, grid_t):.3f} "
        f"FlowGP={physics_residual_rms(flow_samples, grid_t):.3f}"
    )
    print("Held-out test metrics:")
    print(f"  unconstrained GP : RMSE={gp_rmse:.3f}  NLPD={gp_nlpd:.3f}")
    print(f"  FlowGP           : RMSE={flow_rmse:.3f}  NLPD={flow_nlpd:.3f}")

    _plot(grid_t, t_obs, y_obs, gp_samples, flow_samples)


def _panel(ax, title, grid_t, t_obs, y_obs, samples):
    truth = solve_pendulum(np.asarray(grid_t))
    lo, hi = jnp.quantile(samples, jnp.array([0.05, 0.95]), axis=0)
    ax.fill_between(
        grid_t, lo, hi, color="tab:blue", alpha=0.2, label="0.05-0.95 quantiles"
    )
    for s in samples[:15]:
        ax.plot(grid_t, s, color="tab:blue", lw=0.6, alpha=0.5)
    ax.plot(grid_t, truth, color="tab:green", lw=1.5, label="ground truth")
    ax.scatter(t_obs, y_obs, color="red", zorder=5, s=25, label="observations")
    ax.set_title(title)
    ax.set_xlabel("t (s)")
    ax.set_ylim(-2.5, 3.0)


def _plot(grid_t, t_obs, y_obs, gp_samples, flow_samples):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    _panel(axes[0], "Unconstrained GP", grid_t, t_obs, y_obs, gp_samples)
    _panel(axes[1], "FlowGP (ODE-constrained)", grid_t, t_obs, y_obs, flow_samples)
    axes[0].set_ylabel(r"$\theta(t)$")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "pendulum.png")
    fig.savefig(out, dpi=150)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
