"""FlowGP x OT-CFM: a few-step, HMC-usable monotone prior.

This fuses the two methods discussed with Seth:
  * FlowGP samples a *monotone* GP prior -- a distribution OT-CFM/DeepRV cannot
    otherwise sample (no forward generator for "a monotone GP draw").
  * An OT-CFM velocity (the makkunda/dl4bi flow_matching.py recipe: linear
    interpolant x_t = (1-t) z + t f, target velocity v = f - z) is trained on
    those FlowGP samples. The learned velocity is smooth, so the K-step flow is
    HMC-differentiable -- exactly what raw FlowGP is not (fewstep_flowgp_hmc.py:
    acceptance ~0). And because OT paths are ~straight, K=3-5 steps suffice.

Result (honest, and mostly negative for the pipeline):
  * GOOD: the OT-CFM flow is smooth and HMC-differentiable -- HMC acceptance 1.0
    at K=5, where raw FlowGP gave acceptance 0. So OT-CFM does fix the
    HMC-usability problem.
  * BAD: it needs K~10, not 3-5, and it does NOT reproduce the FlowGP monotone
    law -- strict monotone fraction ~0, and the distribution only loosely
    matches even at K=20. Monotonicity is a fragile pointwise constraint that a
    learned-from-samples flow does not preserve (same lesson as the plain
    decoder in deeprv_monotonic.py).

So "FlowGP -> OT-CFM distillation" does not deliver a faithful monotone prior,
which sharpens the real question (see direct_monotone_prior.py): for a shape
constraint you can write a *direct* monotone forward generator and emulate/ use
that with no FlowGP at all. FlowGP's irreplaceable role is not prior emulation
but sampling the data+constraint *posterior*.

Run:
    uv run python replications/flowgp/fm_monotonic.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import optax  # noqa: E402
from deeprv_monotonic import (  # noqa: E402
    M,
    monotone_fraction,
    monotonic_loglik,
    se_kernel,
)
from flowgp import flowgp_sample  # noqa: E402
from hmc_monotonic import hmc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X_GRID = jnp.linspace(0.0, 1.0, M)
K_PRIOR = se_kernel(X_GRID, X_GRID) + 1e-6 * jnp.eye(M)


class FMVelocity(nn.Module):
    """Smooth velocity field v_theta(x_t, t) for OT conditional flow matching."""

    hidden: tuple = (256, 256, 256)

    @nn.compact
    def __call__(self, x, t):
        te = jnp.stack([t, jnp.sin(2 * jnp.pi * t), jnp.cos(2 * jnp.pi * t)], axis=-1)
        h = jnp.concatenate([x, te], axis=-1)
        for hd in self.hidden:
            h = nn.gelu(nn.Dense(hd)(h))
        return nn.Dense(M)(h)


def train_fm(key, f_data, steps=40000, batch=512, lr=1e-3):
    model = FMVelocity()
    params = model.init(key, f_data[:1], jnp.zeros(1))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=1e-2))
    opt_state = opt.init(params)
    n = f_data.shape[0]

    @jax.jit
    def step(params, opt_state, key):
        ki, kz, kt = jax.random.split(key, 3)
        idx = jax.random.randint(ki, (batch,), 0, n)
        f = f_data[idx]
        z = jax.random.normal(kz, f.shape)
        t = jax.random.uniform(kt, (batch,))
        x_t = (1 - t[:, None]) * z + t[:, None] * f  # OT interpolant
        target = f - z  # straight-path velocity

        def loss_fn(p):
            return jnp.mean((model.apply(p, x_t, t) - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    loss = 0.0
    for i in range(steps):
        key, k = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, k)
    print(f"    final OT-CFM loss = {float(loss):.4f}")

    def decode(z, n_steps):
        dt = 1.0 / n_steps

        def st(x, i):
            t = jnp.full((x.shape[0],), (i + 0.5) * dt)
            return x + dt * model.apply(params, x, t), None

        x, _ = jax.lax.scan(st, z, jnp.arange(n_steps))
        return x

    return decode


def main():
    key = jax.random.PRNGKey(0)

    print("[1] FlowGP monotone prior samples as OT-CFM training data ...")
    key, k_data = jax.random.split(key)
    f_data = flowgp_sample(
        k_data, jnp.zeros(M), K_PRIOR, monotonic_loglik, n_samples=20000, T=200, S=1
    )
    f_data.block_until_ready()
    print(
        f"    {f_data.shape[0]} samples, monotone fraction "
        f"{monotone_fraction(f_data):.3f}, pointwise std {float(f_data.std(0).mean()):.2f}"
    )

    print("\n[2] Training OT-CFM velocity on the monotone samples ...")
    key, k_train = jax.random.split(key)
    decode = train_fm(k_train, f_data)

    print("\n[3] Few-step OT-CFM samples (does it stay monotone at small K?):")
    key, k_z = jax.random.split(key)
    z_test = jax.random.normal(k_z, (2000, M))
    ref_mean, ref_std = f_data.mean(0), f_data.std(0)
    for k in [1, 3, 5, 10, 20]:
        s = decode(z_test, k)
        me = float(jnp.sqrt(jnp.mean((s.mean(0) - ref_mean) ** 2)))
        se = float(jnp.sqrt(jnp.mean((s.std(0) - ref_std) ** 2)))
        pos_inc = float(jnp.mean(jnp.diff(s, axis=1) >= 0))  # fraction increasing
        print(
            f"    K={k:2d}:  monotone={monotone_fraction(s):.2f}  "
            f"frac-increasing={pos_inc:.2f}  mean RMSE={me:.2f}  std RMSE={se:.2f}"
        )

    print("\n[4] HMC dose-response with a K=5 OT-CFM flow as the prior ...")
    n_steps = 5

    def decode1(z):
        return decode(z[None, :], n_steps)[0]

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

    samples, acc = hmc(
        k_hmc, potential, jnp.zeros(M), n_samples=2500, step=0.05, n_leap=20
    )
    samples = samples[1000:]
    f_post = jax.vmap(decode1)(samples)
    p_post = jax.nn.sigmoid(f_post)
    rmse = float(
        jnp.sqrt(jnp.mean((p_post.mean(0) - jax.nn.sigmoid(8.0 * (X_GRID - 0.5))) ** 2))
    )
    print(
        f"    acceptance={acc:.2f}  posterior monotone fraction="
        f"{monotone_fraction(f_post):.2f}  recovery RMSE={rmse:.3f}"
    )
    print(
        "    (raw FlowGP in HMC gave acceptance 0; here the smooth OT-CFM flow "
        "works at K=5.)"
    )

    _plot(decode(z_test, 5), x_obs, y_obs, p_post)


def _plot(fm_prior, x_obs, y_obs, p_post):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tr in fm_prior[:15]:
        axes[0].plot(X_GRID, tr, color="tab:blue", lw=0.6, alpha=0.5)
    axes[0].set_title("K=5 OT-CFM samples (trained on FlowGP monotone prior)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    lo, hi = jnp.quantile(p_post, jnp.array([0.05, 0.95]), axis=0)
    axes[1].fill_between(
        X_GRID, lo, hi, color="tab:blue", alpha=0.2, label="posterior 0.05-0.95"
    )
    axes[1].plot(X_GRID, p_post.mean(0), color="tab:blue", label="posterior mean")
    axes[1].plot(
        X_GRID,
        jax.nn.sigmoid(8.0 * (X_GRID - 0.5)),
        "tab:green",
        lw=1.5,
        label="true p(x)",
    )
    axes[1].scatter(x_obs, y_obs, color="red", zorder=5, s=25, label="binary obs")
    axes[1].set_title("HMC dose-response via K=5 OT-CFM prior")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p(x)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "fm_monotonic.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
