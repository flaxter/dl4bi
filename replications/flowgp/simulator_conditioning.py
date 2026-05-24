"""Do agent-based / transmission models fit FlowGP's q(C|f0)? Yes -- a demo.

A complex disease simulator (ABM, network, metapopulation, ...) is exactly a
black box that, given a candidate input function f0, returns outputs you can
*score* against data: q(C | f0) = likelihood of the observed epidemic given the
simulation run on f0. It is non-differentiable (discrete agents, stochastic
events) and stochastic (each run differs) -- so you cannot take gradients and
cannot sample the inverse "which f0 reproduces the data" directly. That is
precisely FlowGP's evaluate-only, gradient-free guidance regime (Eq. 35), and it
is the part of FlowGP that nothing else replaces.

Here f0 = log beta(t), a time-varying transmission rate with a GP prior, and the
black box is a stochastic discrete-time SIR. We infer beta(t) from a noisy daily
incidence curve using ONLY forward simulator runs -- no gradients through the
simulator, no training. This is nonparametric (GP) simulation-based inference of
a simulator's *functional* input, which is the genuinely useful capability for
transmission modelling.

Run:
    uv run python replications/flowgp/simulator_conditioning.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from flowgp import make_schedule, snr_uniform_grid  # noqa: E402
from jax.scipy.stats import poisson  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

DAYS = 60
N_POP = 10000.0
GAMMA = 0.15
I0 = 20.0
M = DAYS
TGRID = jnp.linspace(0.0, 1.0, M)
PRIOR_MEAN = jnp.log(0.2)  # ~ R0 1.3 prior centre
KAPPA, TAU2 = 0.2, 0.7


def se_kernel(xa, xb):
    d = xa[:, None] - xb[None, :]
    return TAU2 * jnp.exp(-(d**2) / (2.0 * KAPPA**2))


def simulate(logbeta, key):
    """Stochastic discrete-time SIR; returns daily incidence (M,). Black box."""
    beta = jnp.exp(logbeta)
    keys = jax.random.split(key, M)

    def day(state, inp):
        s, i = state
        bt, k = inp
        ki, kr = jax.random.split(k)
        lam = bt * i / N_POP
        new_inf = jnp.minimum(jax.random.poisson(ki, s * (1 - jnp.exp(-lam))), s)
        new_rec = jnp.minimum(jax.random.poisson(kr, i * (1 - jnp.exp(-GAMMA))), i)
        return (s - new_inf, i + new_inf - new_rec), new_inf

    _, inc = jax.lax.scan(day, (N_POP - I0, I0), (beta, keys))
    return inc


def loglik(logbeta_flat, key, obs, n_runs=3, temp=20.0):
    """Tempered Poisson fit of obs to the simulator output, averaged over runs.
    Evaluate-only: runs the black box, never differentiates it. logbeta_flat: (B, M)."""

    def one(lb, k):
        inc = jnp.mean(
            jax.vmap(lambda kk: simulate(lb, kk))(jax.random.split(k, n_runs)), axis=0
        )
        return jnp.sum(poisson.logpmf(obs, inc + 0.5)) / temp

    keys = jax.random.split(key, logbeta_flat.shape[0])
    return jax.vmap(one)(logbeta_flat, keys)


def infer_beta(key, K, obs, n_samples=60, T=250, S=48, v_max=40.0, t_min=1e-2):
    """FlowGP gradient-free guidance: sample the GP posterior over log beta(t)
    using only forward simulator scores."""
    beta_s, alpha, log_snr = make_schedule()
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(M))
    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts, alphas, betas = ts[:-1] - ts[1:], alpha(ts[:-1]), beta_s(ts[:-1])
    key, k0 = jax.random.split(key)
    f_hat = jax.random.normal(k0, (n_samples, M))

    def step(carry, inp):
        f_hat, key = carry
        dt, a, b = inp
        key, ke, ks = jax.random.split(key, 3)
        eps = jax.random.normal(ke, (n_samples, S, M))
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps
        logbeta = f0_hat @ L.T + PRIOR_MEAN  # (N,S,M) candidate transmission rates
        sc = loglik(logbeta.reshape(-1, M), ks, obs).reshape(n_samples, S)
        w = jax.nn.softmax(sc, axis=1)
        e_c = jnp.einsum("ns,nsm->nm", w, f0_hat)
        v = 0.5 * b * a / (1.0 - a**2 + 1e-6) * (e_c - a * f_hat)
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(nrm / v_max) / (nrm + 1e-8) * v
        return (f_hat + dt * v, key), None

    (f_hat, _), _ = jax.lax.scan(step, (f_hat, key), (dts, alphas, betas))
    return f_hat @ L.T + PRIOR_MEAN  # posterior log beta(t) samples


def main():
    key = jax.random.PRNGKey(0)
    t = jnp.arange(DAYS)
    true_beta = jnp.where(t < 20, 0.45, jnp.where(t < 40, 0.09, 0.18))

    key, k_obs = jax.random.split(key)
    obs = simulate(jnp.log(true_beta), k_obs)
    print(
        f"observed epidemic: peak {float(obs.max()):.0f} at day {int(jnp.argmax(obs))}, "
        f"total {float(obs.sum()):.0f}/{N_POP:.0f}"
    )

    K = se_kernel(TGRID, TGRID)
    print(
        "inferring beta(t) from the curve via evaluate-only guidance "
        "(forward sim runs only) ..."
    )
    key, k_inf = jax.random.split(key)
    post_logbeta = infer_beta(k_inf, K, obs)
    post_beta = jnp.exp(post_logbeta)

    bmean = post_beta.mean(0)
    rmse = float(jnp.sqrt(jnp.mean((bmean - true_beta) ** 2)))
    print(f"  posterior beta(t): RMSE vs truth = {rmse:.3f}")
    for lo, hi, lbl in [
        (0, 20, "phase1 R0~3"),
        (20, 40, "interv. R0~0.6"),
        (40, 60, "relax R0~1.2"),
    ]:
        bt = float(bmean[lo:hi].mean()) / GAMMA
        print(
            f"    {lbl:18s}: posterior R0 ~ {bt:.2f}  (true "
            f"{float(true_beta[lo:hi].mean()) / GAMMA:.2f})"
        )

    # posterior predictive
    key, k_pp = jax.random.split(key)
    pp = jax.vmap(lambda lb, k: simulate(lb, k))(
        post_logbeta, jax.random.split(k_pp, post_logbeta.shape[0])
    )
    _plot(t, true_beta, post_beta, obs, pp)


def _plot(t, true_beta, post_beta, obs, pp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lo, hi = jnp.quantile(post_beta, jnp.array([0.05, 0.95]), axis=0)
    axes[0].fill_between(
        t, lo, hi, color="tab:blue", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[0].plot(t, post_beta.mean(0), color="tab:blue", label="posterior mean")
    axes[0].plot(t, true_beta, "tab:green", lw=2, label="true beta(t)")
    axes[0].set_title("Recovered transmission rate beta(t)")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("beta(t)")
    axes[0].legend(fontsize=8)

    plo, phi = jnp.quantile(pp, jnp.array([0.05, 0.95]), axis=0)
    axes[1].fill_between(
        t, plo, phi, color="tab:blue", alpha=0.2, label="post. predictive"
    )
    axes[1].plot(t, pp.mean(0), color="tab:blue", label="predictive mean")
    axes[1].scatter(t, obs, color="red", s=12, zorder=5, label="observed incidence")
    axes[1].set_title("Posterior-predictive epidemic curve")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("daily incidence")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Nonparametric GP inference of an ABM input from forward runs only", fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(HERE, "simulator_conditioning.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
