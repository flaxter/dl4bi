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

With ``--llm`` the programmatic ``sitrep_score`` is replaced by an
open-weights instruction-tuned LM (see ``llm_score.LLMVerifier``): each ODE
step renders the candidate trajectories to text and asks the LM
"Is this trajectory consistent with: <SITREP>? Yes/No"; the per-candidate
weights are softmax(log P(Yes) - log P(No)). The closed-form GP posterior on
``D`` is unchanged. Use ``--mock-llm`` to route the JAX verifier through the
same callback path for a no-GPU smoke test.

Run:
    uv run python replications/flowgp/sitrep_fusion.py            # programmatic
    uv run python replications/flowgp/sitrep_fusion.py --mock-llm # mock callback
    uv run python replications/flowgp/sitrep_fusion.py --llm      # real LLM (GPU)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flowgp import make_schedule, snr_uniform_grid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = 80
X = jnp.linspace(0.0, 1.0, M)
KAPPA, TAU2, OBS_STD = 0.07, 2.0, 0.4

# Natural-language sitrep version of `sitrep_score`. NOTE: deliberately
# describes ONLY the late-window resurgence -- the incremental qualitative
# content beyond what the case data D already supports. Mentioning the early
# peak too would double-count (the LLM would reward candidates for a feature
# the data already enforces), per the circularity caveat in HANDOFF.md.
SITREP = (
    "Field surveillance report: in the FINAL portion of the observation window "
    "(roughly the last 25% of the time axis, x near 0.8 to 1.0), incidence has "
    "RISEN AGAIN and is clearly elevated above baseline -- daily case counts in "
    "that late window are well above 1.0. The earlier shape of the outbreak is "
    "not the subject of this report; only the renewed late-period increase."
)


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


def _run_llm_fuse(key, m_pred, K_pred, args):
    """Drive the fuse() loop with a real LLM (or mock) as q(C|f0)."""
    from llm_score import LLMVerifier, MockVerifier, flowgp_pyloop, make_score_fn

    if args.mock_llm:
        backend = MockVerifier(sitrep_score)
        backend.x_grid = np.asarray(X)
    else:
        backend = LLMVerifier(
            model_name=args.model,
            mode=args.mode,
            x_grid=np.asarray(X),
            n_show=args.n_show,
            max_batch=args.batch,
            device=args.device,
        )
    score_fn = make_score_fn(backend, SITREP, temperature=args.temperature)
    fused = flowgp_pyloop(
        key,
        m_pred,
        K_pred,
        score_fn,
        n_samples=args.n_samples,
        T=args.T,
        S=args.S,
        progress=True,
    )
    print(
        f"  [LLM calls: {backend.calls} (~{backend.total_prompts} prompts total)]"
    )
    backend.close()
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=("yesno", "seqlp"), default="yesno")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-show", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

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
    if args.llm or args.mock_llm:
        kind = "LLM" if args.llm else "mock-LLM"
        print(f"\n[fuse + {kind} sitrep score]")
        fused = _run_llm_fuse(k3, m_pred, K_pred, args)
        fused_label = f"D + sitrep ({kind})"
    else:
        fused = fuse(k3, m_pred, K_pred, sitrep_score)
        fused_label = "D + sitrep"

    print(f"\ntrue late level: {float(late_level(truth)):.2f}")
    for name, s in [("D only", d_only), (fused_label, fused)]:
        ll = late_level(s)
        late_rmse = float(
            jnp.sqrt(
                jnp.mean((s[:, int(0.72 * M) :].mean(0) - truth[int(0.72 * M) :]) ** 2)
            )
        )
        print(
            f"  {name:22s}: P(resurgence: late level>0.8)={float(jnp.mean(ll > 0.8)):.2f}"
            f"  mean late level={float(ll.mean()):.2f}  late-window RMSE={late_rmse:.2f}"
        )

    _plot(truth, obs_idx, y, d_only, fused, fused_title=fused_label, tag=args.tag)


def _plot(truth, obs_idx, y, d_only, fused, fused_title="D + sitrep", tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, title, s in [
        (axes[0], "Case data only  p(f|D)", d_only),
        (axes[1], f"{fused_title}  p(f|D)q(C|f)", fused),
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
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(HERE, f"sitrep_fusion{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
