"""The hard case: distilling a PDE-obeying prior with no architectural trick.

Monotonicity and convexity were "easy" wins for distillation: the constraint is
a convex cone that a suitable decoder enforces *by construction*. Physical laws
have no such trick -- you cannot build a decoder that is "pendulum-obeying by
construction". So here the distilled network itself must learn to produce
dynamics-respecting trajectories.

Recipe (same as deeprv_monotonic / convex_emulator):
  1. FlowGP samples a *prior* of damped-pendulum trajectories -- the GP prior
     conditioned only on the ODE  theta'' + sin(theta) + beta theta' = 0  (no
     data). The valid set is essentially a 2-D manifold (one trajectory per
     initial condition), so it is low-dimensional despite living in R^m.
  2. Distil the deterministic flow map (latent = full noise) into a *plain* MLP
     decoder -- no physics baked in.
  3. Test whether fresh-latent emulator draws still obey the ODE (residual), and
     use the emulator as a fast differentiable prior for HMC trajectory
     inference from sparse noisy observations.

Finding: unlike the fragile shape constraints, a plain MLP distils the physics
well -- the low-dimensional solution manifold is smooth and interpolable, so the
emulator's ODE residual stays close to FlowGP's and far below the unconstrained
GP, at ~100x lower cost.

Run:
    uv run python replications/flowgp/pde_emulator.py
"""

from __future__ import annotations

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from convex_emulator import train_decoder  # noqa: E402
from flowgp import flowgp_sample  # noqa: E402
from hmc_monotonic import hmc  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

M = 200
T_H = 30.0  # seconds
BETA = 0.2  # damping
KAPPA = 0.08  # kernel length-scale in normalised time
TAU2 = 1.0
SIGMA_PHYS = 1e-3

T_GRID = jnp.linspace(0.0, T_H, M)
DT = float(T_GRID[1] - T_GRID[0])


def kern(a, b):
    d = a[:, None] - b[None, :]
    return TAU2 * jnp.exp(-(d**2) / (2.0 * KAPPA**2))


def prior_cov():
    tau = T_GRID / T_H
    return kern(tau, tau) + 1e-6 * jnp.eye(M)


def ode_loglik(f0):
    fp = (f0[2:] - f0[:-2]) / (2.0 * DT)
    fpp = (f0[2:] - 2.0 * f0[1:-1] + f0[:-2]) / DT**2
    r = fpp + jnp.sin(f0[1:-1]) + BETA * fp
    return -0.5 * jnp.sum(r**2) / SIGMA_PHYS**2


def resid_rms(s):
    if s.ndim == 1:
        s = s[None, :]
    fp = (s[:, 2:] - s[:, :-2]) / (2.0 * DT)
    fpp = (s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]) / DT**2
    r = fpp + jnp.sin(s[:, 1:-1]) + BETA * fp
    return float(jnp.sqrt(jnp.mean(r**2)))


def flow_prior_pairs(key, K, w):
    n = w.shape[0]
    z = w[:, :M]
    eps = w[:, M:].reshape(n, 1, M)
    return flowgp_sample(
        key, jnp.zeros(M), K, ode_loglik, T=1000, S=1, f_hat_init=z, eps=eps
    )


class Decoder(nn.Module):
    """Plain MLP -- no physics structure baked in."""

    hidden: tuple = (256, 256)
    out_dim: int = M

    @nn.compact
    def __call__(self, w):
        h = w
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        return nn.Dense(self.out_dim)(h)


def true_trajectory(theta0, omega0):
    def rhs(_t, y):
        return [y[1], -np.sin(y[0]) - BETA * y[1]]

    sol = solve_ivp(
        rhs,
        (0.0, T_H),
        [theta0, omega0],
        t_eval=np.asarray(T_GRID),
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    return jnp.asarray(sol.y[0])


def main():
    key = jax.random.PRNGKey(0)
    K = prior_cov()

    print("[1] FlowGP pendulum-ODE-obeying prior vs unconstrained GP prior ...")
    key, k_gp, k_flow = jax.random.split(key, 3)
    L = jnp.linalg.cholesky(K)
    gp_prior = jax.random.normal(k_gp, (500, M)) @ L.T
    flow_prior = flowgp_sample(
        k_flow, jnp.zeros(M), K, ode_loglik, n_samples=500, T=1000, S=5
    )
    print(f"    unconstrained GP residual RMS = {resid_rms(gp_prior):.3f}")
    print(f"    FlowGP ODE-prior residual RMS = {resid_rms(flow_prior):.3f}")

    print("\n[2] Distilling the ODE-obeying prior into a plain MLP ...")
    key, k_w = jax.random.split(key)
    n_train = 10000
    w_train = jax.random.normal(k_w, (n_train, 2 * M))
    t0 = time.time()
    f_train = flow_prior_pairs(key, K, w_train)
    f_train.block_until_ready()
    per_sample_flow = (time.time() - t0) / n_train

    key, k_dec = jax.random.split(key)
    decode, val = train_decoder(k_dec, Decoder(), w_train, f_train, steps=15000)

    key, k_test = jax.random.split(key)
    w_test = jax.random.normal(k_test, (2000, 2 * M))
    emu = decode(w_test)
    emu.block_until_ready()
    t0 = time.time()
    for _ in range(5):
        decode(w_test).block_until_ready()
    per_sample_emu = (time.time() - t0) / (5 * 2000)

    print(f"    distillation val MSE = {val:.4f}")
    print(
        f"    emulator residual RMS (fresh latents) = {resid_rms(emu):.3f}   "
        f"(GP {resid_rms(gp_prior):.3f} -> FlowGP {resid_rms(flow_prior):.3f})"
    )
    print(
        f"    amplitude-decay match: emu std(t0)={float(emu[:, 0].std()):.2f} "
        f"std(tEnd)={float(emu[:, -1].std()):.2f}  vs FlowGP "
        f"{float(flow_prior[:, 0].std()):.2f}/{float(flow_prior[:, -1].std()):.2f}"
    )
    print(
        f"\n[speed] FlowGP {per_sample_flow * 1e3:.2f} ms/sample  vs  "
        f"emulator {per_sample_emu * 1e6:.1f} us/sample  "
        f"-> {per_sample_flow / per_sample_emu:.0f}x faster"
    )

    # [3] HMC trajectory inference with the emulator as a fast physics prior.
    print("\n[3] HMC trajectory inference from sparse noisy data ...")
    truth = true_trajectory(1.5, 0.0)
    key, k_idx, k_y, k_hmc = jax.random.split(key, 4)
    n_obs = 12
    obs_idx = jnp.sort(jax.random.choice(k_idx, M, (n_obs,), replace=False))
    sigma = 0.1
    y_obs = truth[obs_idx] + sigma * jax.random.normal(k_y, (n_obs,))

    def decode1(w):
        return decode(w[None, :])[0]

    def potential(w):
        f = decode1(w)
        ll = -0.5 * jnp.sum((f[obs_idx] - y_obs) ** 2) / sigma**2
        return 0.5 * jnp.sum(w**2) - ll

    w0 = jnp.zeros(2 * M)
    opt = optax.adam(5e-3)
    st = opt.init(w0)

    @jax.jit
    def map_step(w, st):
        g = jax.grad(potential)(w)
        u, st = opt.update(g, st)
        return optax.apply_updates(w, u), st

    for _ in range(3000):
        w0, st = map_step(w0, st)
    samples, acc = hmc(k_hmc, potential, w0, n_samples=3000, step=0.01, n_leap=20)
    samples = samples[1000:]
    f_post = jax.vmap(decode1)(samples)
    rmse = float(jnp.sqrt(jnp.mean((f_post.mean(0) - truth) ** 2)))
    print(
        f"    acceptance={acc:.2f}  posterior residual RMS={resid_rms(f_post):.3f}  "
        f"recovery RMSE vs truth={rmse:.3f}"
    )

    _plot(flow_prior, emu, truth, obs_idx, y_obs, f_post)


def _plot(flow_prior, emu, truth, obs_idx, y_obs, f_post):
    t = np.asarray(T_GRID)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, title, s in [
        (axes[0], "FlowGP ODE prior", flow_prior),
        (axes[1], "Distilled MLP emulator prior", emu),
    ]:
        for tr in s[:15]:
            ax.plot(t, tr, color="tab:blue", lw=0.6, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("t")
    lo, hi = jnp.quantile(f_post, jnp.array([0.05, 0.95]), axis=0)
    axes[2].fill_between(
        t, lo, hi, color="tab:blue", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[2].plot(t, f_post.mean(0), color="tab:blue", label="posterior mean")
    axes[2].plot(t, truth, "tab:green", lw=1.5, label="true trajectory")
    axes[2].scatter(
        t[np.asarray(obs_idx)],
        np.asarray(y_obs),
        color="red",
        s=25,
        zorder=5,
        label="noisy obs",
    )
    axes[2].set_title("HMC trajectory inference (emulator prior)")
    axes[2].set_xlabel("t")
    axes[2].legend(fontsize=8)
    axes[0].set_ylabel(r"$\theta(t)$")
    fig.tight_layout()
    out = os.path.join(HERE, "pde_emulator.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
