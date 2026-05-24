"""Fusing WHO-style natural-language sitreps with outbreak case data.

This is FlowGP's genuinely unique capability: jointly conditioning a GP on
quantitative data D (noisy case counts) AND qualitative text C (a situation
report), via the product-of-experts

    pi(f0 | D, C)  proportional to  p(f0 | D) * q(C | f0).

p(f0 | D) is the closed-form linear-Gaussian GP posterior on the counts;
q(C | f0) is an LLM score of how consistent a candidate trajectory is with the
sitrep -- evaluate-only, so the gradient-free guidance (Eq. 35) applies. Nothing
else fuses these: truncated Gaussians and neural SBI ingest numbers, not text.

Scenario. An outbreak with a main wave and a smaller late resurgence. Case data
is dense early but sparse and noisy in the final weeks, so D alone is *ambiguous*
about whether the late uptick is real. A sitrep then reports "a renewed increase
in cases in the final weeks." The fused posterior should resolve the resurgence
that the data alone leaves uncertain -- text supplying information surveillance
data is too sparse/delayed to pin down, the realistic value-add of sitreps.

(A programmatic verifier stands in for the LLM, since none is available offline;
swap it for an LLM log-likelihood over the trajectory given the sitrep text.)

Run:
    uv run python replications/flowgp/sitrep_fusion.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from flowgp import make_schedule, snr_uniform_grid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = 80
X = jnp.linspace(0.0, 1.0, M)
KAPPA, TAU2, OBS_STD = 0.07, 2.0, 0.4


def kern(a, b):
    d = a[:, None] - b[None, :]
    return TAU2 * jnp.exp(-(d**2) / (2.0 * KAPPA**2))


def true_curve(x):
    # main wave that subsides, then a sustained resurgence in the final weeks
    return 3.0 * jnp.exp(-(((x - 0.30) / 0.10) ** 2)) + 1.6 * jax.nn.sigmoid(
        (x - 0.82) / 0.04
    )


def gp_posterior_on_data(obs_idx, y):
    """Closed-form p(f0 | D): GP regression on noisy case counts."""
    Xo = X[obs_idx]
    Koo = kern(Xo, Xo) + OBS_STD**2 * jnp.eye(obs_idx.size)
    Kgo = kern(X, Xo)
    sol = jnp.linalg.solve(Koo, jnp.eye(obs_idx.size))
    m_pred = Kgo @ sol @ y
    K_pred = kern(X, X) - Kgo @ sol @ Kgo.T
    return m_pred, 0.5 * (K_pred + K_pred.T)


def late_level(f):
    """Resurgence indicator: mean incidence in the final ~12% of the window."""
    return jnp.mean(f[..., int(0.88 * M) :], axis=-1)


def sitrep_score(f):
    """Stand-in LLM likelihood for the sitrep 'cases rising again, elevated in the
    final weeks'. One-sided reward for a sustained late level >~ 1.2.
    Non-differentiable / evaluate-only. f: (..., M)."""
    return -(jnp.maximum(1.2 - late_level(f), 0.0) ** 2) / 0.1


def fuse(
    key, m_pred, K_pred, score, n_samples=300, T=500, S=128, v_max=50.0, t_min=1e-2
):
    """p(f0|D) x q(C|f0) via gradient-free guidance (closed-form D + text C)."""
    beta, alpha, log_snr = make_schedule()
    L = jnp.linalg.cholesky(K_pred + 1e-6 * jnp.eye(M))
    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts, alphas, betas = ts[:-1] - ts[1:], alpha(ts[:-1]), beta(ts[:-1])
    key, k0 = jax.random.split(key)
    f_hat = jax.random.normal(k0, (n_samples, M))

    def step(carry, inp):
        f_hat, key = carry
        dt, a, b = inp
        key, ke = jax.random.split(key)
        eps = jax.random.normal(ke, (n_samples, S, M))
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps
        f0 = f0_hat @ L.T + m_pred
        w = jax.nn.softmax(score(f0), axis=1)
        e_c = jnp.einsum("ns,nsm->nm", w, f0_hat)
        v = 0.5 * b * a / (1.0 - a**2 + 1e-6) * (e_c - a * f_hat)
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(nrm / v_max) / (nrm + 1e-8) * v
        return (f_hat + dt * v, key), None

    (f_hat, _), _ = jax.lax.scan(step, (f_hat, key), (dts, alphas, betas))
    return f_hat @ L.T + m_pred


def main():
    key = jax.random.PRNGKey(0)
    truth = true_curve(X)

    # Case data D: covers the main wave but STOPS before the final weeks
    # (reporting lag) -- so D alone cannot tell a decline from a resurgence there.
    key, k1, k2 = jax.random.split(key, 3)
    obs_idx = jnp.arange(2, int(0.70 * M), 2)
    y = truth[obs_idx] + OBS_STD * jax.random.normal(k1, (obs_idx.size,))

    m_pred, K_pred = gp_posterior_on_data(obs_idx, y)

    # D only
    L = jnp.linalg.cholesky(K_pred + 1e-6 * jnp.eye(M))
    d_only = m_pred + jax.random.normal(k2, (300, M)) @ L.T
    # D + sitrep
    key, k3 = jax.random.split(key)
    fused = fuse(k3, m_pred, K_pred, sitrep_score)

    print(f"true late level: {float(late_level(truth)):.2f}")
    for name, s in [("D only", d_only), ("D + sitrep", fused)]:
        ll = late_level(s)
        late_rmse = float(
            jnp.sqrt(
                jnp.mean((s[:, int(0.72 * M) :].mean(0) - truth[int(0.72 * M) :]) ** 2)
            )
        )
        print(
            f"  {name:12s}: P(resurgence: late level>0.8)={float(jnp.mean(ll > 0.8)):.2f}"
            f"  mean late level={float(ll.mean()):.2f}  late-window RMSE={late_rmse:.2f}"
        )

    _plot(truth, obs_idx, y, d_only, fused)


def _plot(truth, obs_idx, y, d_only, fused):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, title, s in [
        (axes[0], "Case data only  p(f|D)", d_only),
        (axes[1], "Data + sitrep  p(f|D)q(C|f)", fused),
    ]:
        lo, hi = jnp.quantile(s, jnp.array([0.05, 0.95]), axis=0)
        ax.fill_between(X, lo, hi, color="tab:blue", alpha=0.2, label="posterior 5-95%")
        ax.plot(X, s.mean(0), color="tab:blue", label="posterior mean")
        ax.plot(X, truth, "tab:green", lw=1.5, label="true outbreak")
        ax.scatter(X[obs_idx], y, color="red", s=18, zorder=5, label="case data")
        ax.axvspan(0.72, 1.0, color="tab:orange", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("incidence")
    fig.suptitle(
        "Sitrep 'renewed increase in the final weeks' resolves the "
        "data-ambiguous resurgence",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "sitrep_fusion.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
