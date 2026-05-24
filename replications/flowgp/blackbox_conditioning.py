"""Conditioning a GP on a black-box / text-style likelihood (FlowGP's real niche).

This is the one regime where FlowGP is irreplaceable: a constraint C that is
nonlinear, non-generatable, and only *evaluable* -- e.g. an LLM scoring how well a
function matches a natural-language description (paper Section 6.3). There is no
truncated-Gaussian form and no forward generator; all you can do is, given a
candidate function f0, get back a scalar score q(C | f0).

Mechanism (gradient-free guidance, paper Appendix C / Eq. 35). The guidance term
needs only the *difference of conditional means*

    grad_{f_t} log q(C | f_t)  =  alpha/(1-alpha^2) * ( E[f0 | f_t, C] - E[f0 | f_t] ),

and E[f0 | f_t, C] is an importance-weighted average of candidate draws
f0^(i) ~ p(f0 | f_t), with weights w^(i) proportional to the black-box score
q(C | f0^(i)). So each ODE step only *calls* the scorer on S candidates -- no
gradients, no backprop through the LLM. (With a real LLM, q(C|f0) is e.g. the
log-prob it assigns to the tokenised function values given the prompt, a la
Biggs & Willis; here we use a programmatic verifier as a stand-in, since no LLM
is available offline.)

The verifier below encodes the "description": a single smooth bump, ~0 at both
ends, peaking near x=0.7 at height ~2. It uses argmax / mode-counting / thresholds
-- non-differentiable and non-generatable -- exactly the setting that defeats
truncated Gaussians, direct generators, and gradient guidance, but that FlowGP's
evaluate-only guidance handles.

With ``--llm`` the programmatic verifier is replaced by a real open-weights
instruction-tuned LM (Qwen/Llama via ``llm_score.LLMVerifier``) that scores
each candidate trajectory by ``log P(Yes) - log P(No)`` to the prompt
"Is this function consistent with: <DESCRIPTION>? Yes/No". The guidance loop
is unchanged -- only the score function is swapped (paper Sec 6.3, Eq. 35).
Use ``--mock-llm`` to route the programmatic verifier through the same callback
path for smoke-testing without torch/GPU.

Run:
    uv run python replications/flowgp/blackbox_conditioning.py            # programmatic
    uv run python replications/flowgp/blackbox_conditioning.py --mock-llm # mock callback
    uv run python replications/flowgp/blackbox_conditioning.py --llm      # real LLM (GPU)
"""

from __future__ import annotations

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flowgp import make_schedule, snr_uniform_grid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = 64
X_GRID = jnp.linspace(0.0, 1.0, M)
LENGTHSCALE = 0.25  # smooth enough that a single bump is prior-plausible

# "Description" targets (what an LLM prompt would convey).
PEAK_LOC, PEAK_VAL = 0.7, 2.0

# Natural-language version of the same description, for the LLM verifier.
DESCRIPTION = (
    "a single smooth bump on the interval [0, 1], starting and ending near "
    "zero at the endpoints, rising to a peak near x=0.7 with maximum value "
    "close to +2.0, and otherwise monotone on either side of the peak."
)


def se_kernel(xa, xb):
    d = xa[:, None] - xb[None, :]
    return jnp.exp(-(d**2) / (2.0 * LENGTHSCALE**2))


def describe_logscore(f):
    """Black-box verifier standing in for an LLM: scores how well f matches
    'a single smooth bump, ~0 at the ends, peaking near x=0.7 at height ~2'.
    Uses argmax / mode counting / thresholds -- evaluate-only, non-differentiable.
    f: (..., M)."""
    peak_idx = jnp.argmax(f, axis=-1)
    peak_loc = X_GRID[peak_idx]
    peak_val = jnp.max(f, axis=-1)
    d = jnp.diff(f, axis=-1)
    sign = jnp.sign(d)
    n_max = jnp.sum(
        (sign[..., :-1] > 0) & (sign[..., 1:] < 0), axis=-1
    )  # interior maxima
    return -(
        (peak_loc - PEAK_LOC) ** 2 / 0.01
        + (peak_val - PEAK_VAL) ** 2 / 0.5
        + f[..., 0] ** 2 / 0.05
        + f[..., -1] ** 2 / 0.05
        + jnp.maximum(n_max - 1, 0) * 5.0  # unimodality
    )


def blackbox_guided_sample(
    key,
    m_pred,
    K_pred,
    logscore,
    n_samples,
    T=600,
    S=128,
    v_max=50.0,
    t_min=1e-2,
    jitter=1e-6,
):
    """FlowGP with *gradient-free* (Fisher-identity) guidance: each step only
    evaluates ``logscore`` on S candidates -- never differentiates it."""
    beta, alpha, log_snr = make_schedule()
    L = jnp.linalg.cholesky(K_pred + jitter * jnp.eye(M))
    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts, alphas, betas = ts[:-1] - ts[1:], alpha(ts[:-1]), beta(ts[:-1])

    key, k0 = jax.random.split(key)
    f_hat = jax.random.normal(k0, (n_samples, M))

    def step(carry, inp):
        f_hat, key = carry
        dt, a, b = inp
        key, ke = jax.random.split(key)
        eps = jax.random.normal(ke, (n_samples, S, M))
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps  # (N,S,M)
        f0 = f0_hat @ L.T + m_pred
        w = jax.nn.softmax(logscore(f0), axis=1)  # (N,S) black-box, evaluate-only
        e_c = jnp.einsum("ns,nsm->nm", w, f0_hat)  # E[f0_hat | f_t, C]
        e_0 = a * f_hat  # E[f0_hat | f_t]
        v = 0.5 * b * a / (1.0 - a**2 + 1e-6) * (e_c - e_0)  # toward the C-mean
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(nrm / v_max) / (nrm + 1e-8) * v
        return (f_hat + dt * v, key), None

    (f_hat, _), _ = jax.lax.scan(step, (f_hat, key), (dts, alphas, betas))
    return f_hat @ L.T + m_pred


def _report(name, s):
    pk = X_GRID[jnp.argmax(s, axis=1)]
    pv = jnp.max(s, axis=1)
    sign = jnp.sign(jnp.diff(s, axis=1))
    nmax = jnp.sum((sign[:, :-1] > 0) & (sign[:, 1:] < 0), axis=1)
    print(
        f"  {name:14s}: peak loc {float(pk.mean()):.2f}+-{float(pk.std()):.2f} "
        f"(target {PEAK_LOC}) | peak val {float(pv.mean()):.2f}+-{float(pv.std()):.2f} "
        f"(target {PEAK_VAL}) | unimodal frac {float(jnp.mean(nmax == 1)):.2f} | "
        f"|f(0)|+|f(1)| {float((jnp.abs(s[:, 0]) + jnp.abs(s[:, -1])).mean()):.2f}"
    )


def _run_llm(key, m_pred, K_pred, args):
    """Drive FlowGP with a real LLM (or mock) as the verifier."""
    from llm_score import LLMVerifier, MockVerifier, flowgp_pyloop, make_score_fn

    if args.mock_llm:
        backend = MockVerifier(describe_logscore)
        backend.x_grid = np.asarray(X_GRID)  # for debug_show_prompt
    else:
        backend = LLMVerifier(
            model_name=args.model,
            mode=args.mode,
            x_grid=np.asarray(X_GRID),
            n_show=args.n_show,
            max_batch=args.batch,
            device=args.device,
        )
    score_fn = make_score_fn(backend, DESCRIPTION, temperature=args.temperature)
    cond = flowgp_pyloop(
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
    return cond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use a real LLM verifier")
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="route the JAX programmatic verifier through the LLM-callback path",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=("yesno", "seqlp"), default="yesno")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-show", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="divides the LLM log-score before the softmax; raise to avoid weight collapse",
    )
    parser.add_argument("--tag", default="", help="suffix on the output figure filename")
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    K = se_kernel(X_GRID, X_GRID) + 1e-6 * jnp.eye(M)

    print("[prior] unconditional GP samples (no notion of the description)")
    key, k_p = jax.random.split(key)
    L = jnp.linalg.cholesky(K)
    prior = jax.random.normal(k_p, (200, M)) @ L.T

    if args.llm or args.mock_llm:
        kind = "LLM" if args.llm else "mock-LLM"
        print(f"\n[FlowGP+{kind}] conditioning via gradient-free guidance through llm_score")
        key, k_f = jax.random.split(key)
        cond = _run_llm(k_f, jnp.zeros(M), K, args)
        label = f"FlowGP+{kind}"
    else:
        print(
            "\n[FlowGP] conditioning on the programmatic verifier via evaluate-only guidance"
        )
        key, k_f = jax.random.split(key)
        cond = blackbox_guided_sample(k_f, jnp.zeros(M), K, describe_logscore, 200)
        label = "FlowGP"

    _report("prior", prior)
    _report(label, cond)
    print(
        "\n  The GP is steered to match a description it can only be *scored* "
        "against --\n  no gradient, no generator. That is FlowGP's irreducible niche; "
        "with --llm\n  the verifier is an open-weights instruct LM (Section 6.3)."
    )

    _plot(prior, cond, label=label, tag=args.tag)


def _plot(prior, cond, label="FlowGP", tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    axes[0].set_title("Unconditional GP prior")
    for tr in prior[:25]:
        axes[0].plot(X_GRID, tr, color="tab:gray", lw=0.5, alpha=0.5)
    axes[1].set_title(f"Conditioned via {label} (evaluate-only)")
    n_draw = min(25, cond.shape[0])
    for tr in cond[:n_draw]:
        axes[1].plot(X_GRID, tr, color="tab:blue", lw=0.6, alpha=0.6)
    axes[1].plot(X_GRID, cond.mean(0), color="tab:blue", lw=2, label="mean")
    axes[1].scatter(
        [PEAK_LOC],
        [PEAK_VAL],
        color="red",
        zorder=5,
        s=60,
        marker="*",
        label="described peak (0.7, 2)",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    fig.suptitle(
        "FlowGP conditions a GP on a non-differentiable, non-generatable description",
        fontsize=11,
    )
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(HERE, f"blackbox_conditioning{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
