"""Does the FlowGP step matter for a physics prior? A direct-simulation baseline.

Seth's question: for the pendulum ODE-obeying prior (pde_emulator.py), do we need
FlowGP at all -- couldn't we train the emulator directly from simulated data?

Answer, demonstrated here: for a *forward-simulable* prior, FlowGP is unnecessary
and inferior. The damped-pendulum solution set is a 2-parameter family (one
trajectory per initial condition), so we can sample it *exactly* by integrating
the ODE from random ICs with a differentiable RK4 solver. That solver drops
straight into HMC over a 2-D latent -- lower ODE residual, faster, simpler, and
with no GP/FlowGP/distillation anywhere.

Compare against pde_emulator.py (FlowGP -> distilled MLP):
    FlowGP prior residual ~0.09, distilled-emulator residual ~0.19,
    generation ~8.5 ms/sample, latent dim 2M=400, HMC recovery RMSE ~0.05.

FlowGP earns its keep only where there is *no* forward generator -- shape
constraints like monotonicity/convexity -- or, the paper's real contribution,
when sampling the *posterior* that fuses observed data with a nonlinear
constraint p(C | f0). A forward simulator gives you the prior for free; it does
not give you that posterior.

Run:
    uv run python replications/flowgp/direct_sim_baseline.py
"""

from __future__ import annotations

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from hmc_monotonic import hmc  # noqa: E402
from pde_emulator import (  # noqa: E402
    BETA,
    DT,
    T_GRID,
    M,
    ode_loglik,
    prior_cov,
    resid_rms,
    true_trajectory,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA_THETA, SIGMA_OMEGA = 2.0, 1.0  # initial-condition prior scales


def rk4_traj(theta0, omega0, n_sub=4):
    """Differentiable RK4 integration of the damped pendulum, sampled on T_GRID."""
    dt = DT / n_sub

    def deriv(s):
        return jnp.array([s[1], -jnp.sin(s[0]) - BETA * s[1]])

    def grid_step(s, _):
        def sub(ss, _):
            k1 = deriv(ss)
            k2 = deriv(ss + 0.5 * dt * k1)
            k3 = deriv(ss + 0.5 * dt * k2)
            k4 = deriv(ss + dt * k3)
            return ss + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

        s_next, _ = jax.lax.scan(sub, s, None, length=n_sub)
        return s_next, s_next[0]

    s0 = jnp.array([theta0, omega0])
    _, theta_rest = jax.lax.scan(grid_step, s0, None, length=M - 1)
    return jnp.concatenate([jnp.array([theta0]), theta_rest])


integrate = jax.jit(jax.vmap(rk4_traj))


def main():
    key = jax.random.PRNGKey(0)

    # --- Direct-simulation prior: sample ICs, integrate the ODE exactly. -------
    print("[1] Direct-simulation ODE prior (no FlowGP, no GP) ...")
    key, k_ic = jax.random.split(key)
    n = 10000
    ic = jax.random.normal(k_ic, (n, 2))
    theta0 = SIGMA_THETA * ic[:, 0]
    omega0 = SIGMA_OMEGA * ic[:, 1]
    integrate(theta0[:8], omega0[:8]).block_until_ready()  # warm up jit
    t0 = time.time()
    trajs = integrate(theta0, omega0)
    trajs.block_until_ready()
    per_sample = (time.time() - t0) / n
    print(
        f"    {n} exact trajectories in {per_sample * n:.2f}s "
        f"({per_sample * 1e6:.1f} us/sample)"
    )
    print(f"    direct-sim residual RMS = {resid_rms(trajs):.4f}")

    # FlowGP prior residual for reference (small batch is enough).
    key, k_flow = jax.random.split(key)
    K = prior_cov()
    flow_prior = flowgp_sample(
        k_flow, jnp.zeros(M), K, ode_loglik, n_samples=200, T=1000, S=5
    )
    print(
        f"    FlowGP residual RMS (ref) = {resid_rms(flow_prior):.4f}  "
        f"(~8.5 ms/sample, 400-dim latent)"
    )

    # --- HMC inference using the differentiable solver directly as the prior. --
    print("\n[2] HMC trajectory inference with the differentiable ODE solver ...")
    truth = true_trajectory(1.5, 0.0)
    key, k_idx, k_y, k_hmc = jax.random.split(key, 4)
    n_obs = 12
    obs_idx = jnp.sort(jax.random.choice(k_idx, M, (n_obs,), replace=False))
    sigma = 0.1
    y_obs = truth[obs_idx] + sigma * jax.random.normal(k_y, (n_obs,))

    def potential(w):  # w in R^2 ~ N(0, I); ICs = scaled w
        traj = rk4_traj(SIGMA_THETA * w[0], SIGMA_OMEGA * w[1])
        ll = -0.5 * jnp.sum((traj[obs_idx] - y_obs) ** 2) / sigma**2
        return 0.5 * jnp.sum(w**2) - ll

    # The IC-parameterised posterior is stiff: a pendulum trajectory is sensitive
    # to its ICs over 30 s, so the likelihood surface is sharp. We find the mode
    # with multi-start gradient descent, then run a small-step HMC around it.
    import optax

    key, k_starts = jax.random.split(key)
    starts = 1.5 * jax.random.normal(k_starts, (40, 2))

    def descend(w):
        opt = optax.adam(2e-2)
        st = opt.init(w)

        def body(_, carry):
            w, st = carry
            g = jax.grad(potential)(w)
            u, st = opt.update(g, st)
            return optax.apply_updates(w, u), st

        w, _ = jax.lax.fori_loop(0, 800, body, (w, st))
        return w, potential(w)

    cand, vals = jax.vmap(descend)(starts)
    w_map = cand[jnp.argmin(vals)]

    samples, acc = hmc(k_hmc, potential, w_map, n_samples=4000, step=0.004, n_leap=25)
    samples = samples[1000:]
    f_post = jax.vmap(lambda w: rk4_traj(SIGMA_THETA * w[0], SIGMA_OMEGA * w[1]))(
        samples
    )
    rmse = float(jnp.sqrt(jnp.mean((f_post.mean(0) - truth) ** 2)))
    print(f"    latent dim = 2   acceptance = {acc:.2f}")
    print(
        f"    posterior residual RMS = {resid_rms(f_post):.4f}  "
        f"recovery RMSE vs truth = {rmse:.3f}"
    )
    print(
        "\n  Verdict: for a forward-simulable prior the FlowGP+distillation"
        " pipeline is\n  redundant -- direct simulation is exact, faster, and"
        " uses a 2-D latent.\n  FlowGP matters when no forward generator exists"
        " (shape constraints) or for\n  the data+constraint *posterior*, which"
        " has no simulator."
    )

    _plot(trajs, truth, obs_idx, y_obs, f_post)


def _plot(trajs, truth, obs_idx, y_obs, f_post):
    t = np.asarray(T_GRID)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for tr in trajs[:15]:
        axes[0].plot(t, tr, color="tab:purple", lw=0.6, alpha=0.5)
    axes[0].set_title("Direct-simulation ODE prior (no FlowGP)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\theta(t)$")

    lo, hi = jnp.quantile(f_post, jnp.array([0.05, 0.95]), axis=0)
    axes[1].fill_between(
        t, lo, hi, color="tab:purple", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[1].plot(t, f_post.mean(0), color="tab:purple", label="posterior mean")
    axes[1].plot(t, truth, "tab:green", lw=1.5, label="true trajectory")
    axes[1].scatter(
        t[np.asarray(obs_idx)],
        np.asarray(y_obs),
        color="red",
        s=25,
        zorder=5,
        label="noisy obs",
    )
    axes[1].set_title("HMC inference via differentiable solver (2-D latent)")
    axes[1].set_xlabel("t")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "direct_sim_baseline.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
