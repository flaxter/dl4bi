"""Physics-informed GP regression: viscous Burgers' equation (Appendix H.4).

Reproduces the Burgers' panel of the FlowGP paper (arXiv:2605.21041). A GP is
conditioned only on (noisy) observations of the *initial condition* at t = 0,
and FlowGP must extrapolate the full space-time field forward in time by
enforcing the PDE

    u_t + u u_x = nu u_xx,    nu = 0.02,   x in [-1, 1], t in [0, 1],

together with homogeneous Dirichlet boundaries u(+-1, t) = 0. Both are encoded
as Gaussian likelihoods on finite-difference residuals (Eqs. 77-80).

The kernel hyperparameters are fixed to the benchmark values of Chen et al.
(kappa_x = 0.025, kappa_t = 0.3, tau^2 = 1.0). Ground truth comes from our own
fine-grid solver since the reference data is not public. As the paper notes,
Burgers' uniquely needs ~10,000 ODE steps to resolve the flow.

Run:
    uv run python replications/flowgp/burgers.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from gp_utils import extend_to_test, gp_posterior, predictive_metrics  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

NU = 0.02  # kinematic viscosity
H, W = 50, 20  # spatial x temporal grid -> m = 1000
KAPPA_X, KAPPA_T = 0.025, 0.3
TAU2 = 1.0
N_OBS = 100  # dense, near-noiseless initial-condition observations
OBS_NOISE_STD = 1e-6
SIGMA_PHYS = 1e-5
SIGMA_BC = 1e-6
T_STEPS = 10000
N_SAMPLES = 30
TEST_TIMES = (0.2, 0.5, 0.8)


def initial_condition(x):
    return -np.sin(np.pi * (2.0 * x - 1.0))


def solve_burgers(x_query, t_query, n_fine=512):
    """Reference solution via method of lines on a fine spatial grid.

    Returns an array of shape (len(x_query), len(t_query)).
    """
    xf = np.linspace(-1.0, 1.0, n_fine)
    dx = xf[1] - xf[0]
    u0 = initial_condition(xf)
    u0[0] = u0[-1] = 0.0

    def rhs(_t, u):
        ux = np.zeros_like(u)
        uxx = np.zeros_like(u)
        ux[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        du = -u * ux + NU * uxx
        du[0] = du[-1] = 0.0
        return du

    sol = solve_ivp(
        rhs,
        (0.0, max(t_query)),
        u0,
        t_eval=t_query,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )
    # Interpolate each time snapshot onto the requested spatial points.
    out = np.empty((len(x_query), len(t_query)))
    for j in range(len(t_query)):
        out[:, j] = np.interp(x_query, xf, sol.y[:, j])
    return out


def make_grid():
    xs = jnp.linspace(-1.0, 1.0, H)
    ts = jnp.linspace(0.0, 1.0, W)
    XX, TT = jnp.meshgrid(xs, ts, indexing="ij")  # (H, W); flat index = i*W + j
    X_grid = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
    return xs, ts, X_grid


def make_physics_loglik():
    dx = 2.0 / (H - 1)
    dt = 1.0 / (W - 1)

    def loglik(f0):
        F = f0.reshape(H, W)
        dfdt = (F[:, 2:] - F[:, :-2]) / (2.0 * dt)  # (H, W-2)
        dfdx = (F[2:, :] - F[:-2, :]) / (2.0 * dx)  # (H-2, W)
        d2fdx2 = (F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]) / dx**2
        r = (
            dfdt[1:-1, :] + F[1:-1, 1:-1] * dfdx[:, 1:-1] - NU * d2fdx2[:, 1:-1]
        )  # interior residual (Eq. 80), shape (H-2, W-2)
        bc = jnp.concatenate([F[0, :], F[-1, :]])  # Dirichlet rows
        return -0.5 * jnp.sum(r**2) / SIGMA_PHYS**2 - 0.5 * jnp.sum(bc**2) / SIGMA_BC**2

    return loglik


def pde_residual_rms(samples):
    dx = 2.0 / (H - 1)
    dt = 1.0 / (W - 1)
    F = samples.reshape(-1, H, W)
    dfdt = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dt)
    dfdx = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dx)
    d2fdx2 = (F[:, 2:, :] - 2.0 * F[:, 1:-1, :] + F[:, :-2, :]) / dx**2
    r = dfdt[:, 1:-1, :] + F[:, 1:-1, 1:-1] * dfdx[:, :, 1:-1] - NU * d2fdx2[:, :, 1:-1]
    return float(jnp.sqrt(jnp.mean(r**2)))


def fixed_params():
    return {
        "log_ls": jnp.log(jnp.array([KAPPA_X, KAPPA_T])),
        "log_os": jnp.log(jnp.array(TAU2)),
        "a": jnp.array(0.0),
        "b": jnp.zeros(2),
        "noise_var": OBS_NOISE_STD**2,
    }


def main():
    rng = np.random.default_rng(0)
    xs, ts, X_grid = make_grid()

    # Initial-condition observations at t = 0 (interior x points only).
    x_obs = np.linspace(-1.0, 1.0, N_OBS + 2)[1:-1]
    y_obs = initial_condition(x_obs) + rng.normal(0.0, OBS_NOISE_STD, size=N_OBS)
    Xo = jnp.stack([jnp.asarray(x_obs), jnp.zeros(N_OBS)], axis=1)
    yo = jnp.asarray(y_obs)

    params = fixed_params()
    mean_fn, cov_fn = gp_posterior(Xo, yo, params)
    m_pred = mean_fn(X_grid)
    K_pred = 0.5 * (cov_fn(X_grid, X_grid) + cov_fn(X_grid, X_grid).T)

    loglik = make_physics_loglik()

    key = jax.random.PRNGKey(1)
    key, k_gp = jax.random.split(key)
    Lg = jnp.linalg.cholesky(K_pred + 1e-8 * jnp.eye(H * W))
    gp_samples = m_pred + jax.random.normal(k_gp, (N_SAMPLES, H * W)) @ Lg.T

    key, k_flow = jax.random.split(key)
    flow_samples = flowgp_sample(
        k_flow, m_pred, K_pred, loglik, n_samples=N_SAMPLES, T=T_STEPS, S=5
    )

    # Test points: every spatial grid x at the three evaluation times.
    x_np = np.asarray(xs)
    truth = solve_burgers(x_np, list(TEST_TIMES))  # (H, 3)
    test_coords = jnp.stack(
        [jnp.repeat(xs, len(TEST_TIMES)), jnp.tile(jnp.asarray(TEST_TIMES), H)],
        axis=1,
    )
    y_test = jnp.asarray(truth.ravel())  # ordering matches test_coords (x major)

    gp_test = extend_to_test(
        gp_samples, X_grid, test_coords, m_pred, K_pred, mean_fn, cov_fn
    )
    flow_test = extend_to_test(
        flow_samples, X_grid, test_coords, m_pred, K_pred, mean_fn, cov_fn
    )
    gp_rmse, gp_nlpd = predictive_metrics(gp_test, y_test, OBS_NOISE_STD**2)
    flow_rmse, flow_nlpd = predictive_metrics(flow_test, y_test, OBS_NOISE_STD**2)

    print(
        f"PDE residual RMS (grid): GP={pde_residual_rms(gp_samples):.3f} "
        f"FlowGP={pde_residual_rms(flow_samples):.3f}"
    )
    print("Held-out test metrics (pooled over t in {0.2,0.5,0.8}):")
    print(f"  unconstrained GP : RMSE={gp_rmse:.3f}  NLPD={gp_nlpd:.3f}")
    print(f"  FlowGP           : RMSE={flow_rmse:.3f}  NLPD={flow_nlpd:.3f}")

    _plot(x_np, truth, gp_test, flow_test)


def _slice(test_samples, j):
    """Extract predictions for test time index j (test_coords are x-major)."""
    return test_samples[:, j :: len(TEST_TIMES)]  # (N_SAMPLES, H)


def _plot(x_np, truth, gp_test, flow_test):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for col, t in enumerate(TEST_TIMES):
        for row, (name, samp) in enumerate([("GP", gp_test), ("FlowGP", flow_test)]):
            ax = axes[row, col]
            s = _slice(samp, col)
            lo, hi = jnp.quantile(s, jnp.array([0.05, 0.95]), axis=0)
            ax.fill_between(x_np, lo, hi, color="tab:blue", alpha=0.2)
            ax.plot(x_np, jnp.mean(s, axis=0), color="tab:blue", label=f"{name} mean")
            ax.plot(x_np, truth[:, col], "tab:green", lw=1.5, label="truth")
            if row == 0:
                ax.set_title(f"t = {t}")
            if col == 0:
                ax.set_ylabel(f"{name}\nu(x, t)")
            ax.set_xlabel("x")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "burgers.png")
    fig.savefig(out, dpi=150)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
