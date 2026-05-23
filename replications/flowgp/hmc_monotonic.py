"""Drop-in HMC with a FlowGP-distilled monotonic prior.

End-to-end demonstration of the payoff from ``deeprv_monotonic.py``: a monotone
GP prior, distilled from FlowGP into a fast differentiable decoder, used as a
non-centred prior inside Hamiltonian Monte Carlo for a model with a non-Gaussian
likelihood (Bernoulli dose-response).

The latent is w ~ N(0, I); the monotone latent function is f = decode(w); and
the observations are binary y_i ~ Bernoulli(sigmoid(f(x_i))). HMC explores w
using gradients through the decoder -- exactly the role DeepRV plays for an
unconstrained GP, but here every posterior draw is monotone by construction.

A hand-written HMC keeps this dependency-free (no NumPyro). It is a proof of
concept, not a tuned sampler.

Run:
    uv run python replications/flowgp/hmc_monotonic.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from deeprv_monotonic import (  # noqa: E402
    M,
    flow_prior_pairs,
    monotone_fraction,
    se_kernel,
    train_decoder,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def hmc(key, potential, q0, n_samples, step, n_leap):
    """Minimal Hamiltonian Monte Carlo with a unit mass matrix."""
    grad_U = jax.grad(potential)

    def leapfrog(q, p):
        p = p - 0.5 * step * grad_U(q)

        def body(_, qp):
            q, p = qp
            q = q + step * p
            p = p - step * grad_U(q)
            return (q, p)

        q, p = jax.lax.fori_loop(0, n_leap - 1, body, (q, p))
        q = q + step * p
        p = p - 0.5 * step * grad_U(q)
        return q, p

    def stepper(carry, k):
        q, _ = carry
        kp, ka = jax.random.split(k)
        p = jax.random.normal(kp, q.shape)
        q_new, p_new = leapfrog(q, p)
        h0 = potential(q) + 0.5 * jnp.sum(p**2)
        h1 = potential(q_new) + 0.5 * jnp.sum(p_new**2)
        accept = jnp.log(jax.random.uniform(ka)) < (h0 - h1)
        q = jnp.where(accept, q_new, q)
        return (q, accept), (q, accept)

    keys = jax.random.split(key, n_samples)
    (_, _), (qs, accepts) = jax.lax.scan(stepper, (q0, False), keys)
    return qs, float(jnp.mean(accepts))


def main():
    key = jax.random.PRNGKey(0)
    x_grid = jnp.linspace(0.0, 1.0, M)
    K = se_kernel(x_grid, x_grid) + 1e-6 * jnp.eye(M)

    # Train the distilled monotone emulator (smaller set; this is just a demo).
    print("Training FlowGP-distilled monotone emulator ...")
    key, k_w = jax.random.split(key)
    w_train = jax.random.normal(k_w, (6000, 2 * M))
    f_train = flow_prior_pairs(key, K, w_train)
    key, k_dec = jax.random.split(key)
    decode = train_decoder(k_dec, w_train, f_train, steps=15000)

    def decode1(w):
        return decode(w[None, :])[0]

    # Synthetic dose-response data: monotone true log-odds, binary outcomes.
    key, k_x, k_y = jax.random.split(key, 3)
    n_obs = 25
    x_obs = jnp.sort(jax.random.uniform(k_x, (n_obs,)))
    true_logodds = 8.0 * (x_obs - 0.5)
    y_obs = (jax.random.uniform(k_y, (n_obs,)) < jax.nn.sigmoid(true_logodds)).astype(
        jnp.float64
    )
    obs_idx = jnp.clip(jnp.round(x_obs * (M - 1)).astype(int), 0, M - 1)

    def potential(w):
        f = decode1(w)
        logits = f[obs_idx]
        log_lik = jnp.sum(
            y_obs * jax.nn.log_sigmoid(logits)
            + (1.0 - y_obs) * jax.nn.log_sigmoid(-logits)
        )
        return 0.5 * jnp.sum(w**2) - log_lik  # -log posterior (up to const)

    print("Running HMC over the emulator latent ...")
    key, k_hmc = jax.random.split(key)
    q0 = jnp.zeros(2 * M)
    samples, acc = hmc(k_hmc, potential, q0, n_samples=3000, step=0.03, n_leap=25)
    samples = samples[1000:]  # burn-in
    print(f"    acceptance rate = {acc:.2f}  ({samples.shape[0]} post-burn-in draws)")

    f_post = jax.vmap(decode1)(samples)
    p_post = jax.nn.sigmoid(f_post)
    print(f"    posterior-sample monotone fraction = {monotone_fraction(f_post):.3f}")
    rmse = float(
        jnp.sqrt(jnp.mean((p_post.mean(0) - jax.nn.sigmoid(8.0 * (x_grid - 0.5))) ** 2))
    )
    print(f"    posterior-mean p(x) RMSE vs truth = {rmse:.3f}")

    _plot(x_grid, x_obs, y_obs, p_post)


def _plot(x_grid, x_obs, y_obs, p_post):
    lo, hi = jnp.quantile(p_post, jnp.array([0.05, 0.95]), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(
        x_grid, lo, hi, color="tab:blue", alpha=0.2, label="posterior 0.05-0.95"
    )
    ax.plot(x_grid, p_post.mean(0), color="tab:blue", label="posterior mean")
    ax.plot(
        x_grid,
        jax.nn.sigmoid(8.0 * (x_grid - 0.5)),
        "tab:green",
        lw=1.5,
        label="true p(x)",
    )
    ax.scatter(x_obs, y_obs, color="red", zorder=5, s=25, label="binary obs")
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax.set_title("Monotone dose-response via HMC + FlowGP-distilled prior")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "hmc_monotonic.png")
    fig.savefig(out, dpi=150)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
