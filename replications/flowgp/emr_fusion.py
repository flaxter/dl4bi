"""EMR three-factor fusion: lab values + ICD/billing codes + clinical notes.

The killer-app extension of `sitrep_fusion.py`: one latent patient trajectory
`f` jointly conditioned on **three** heterogeneous data streams via a
product-of-experts

    pi(f | D, C_codes, C_text)
        proportional to  p(f | D)  *  q(C_codes | f)  *  q(C_text | f).

p(f | D) is the closed-form Gaussian GP posterior on noisy lab values (the
same closed-form linear-Gaussian conditioning as before). The two non-Gaussian
factors are evaluate-only -- a structured score for ICD/billing codes
(threshold crossings, durations, treatment effects) and an LLM score for the
free-text clinical note. Because all three terms multiply, their *log-scores
add*; FlowGP's softmax over S candidates handles a sum of log-scores just as
naturally as a single one. So `q_total(f) = q_codes(f) + q_text(f)` is the
ONLY change to the sampler -- per-factor temperatures are easy to tune
independently.

This is the irreducible niche of FlowGP made fully multimodal: nothing else
fuses sparse numbers, structured codes, and free text into a single coherent
posterior over a latent trajectory.

Scenario. A 14-day inpatient stay. Latent f(t) is a "severity index". Streams:

    D       : noisy nurse-recorded labs at days 1, 3, 5, 7 only (sparse).
    codes   : ICD R50.9 ("fever") AND Z51.11 ("antibiotic given on day 6").
              -> q_codes rewards f > 1.5 for >=1 day during days 1..6 AND
                 a downward trend in days 6..10.
    notes   : "Days 11-13: subjective worsening, mild fever returning."
              -> q_text is an LLM verifier on that paragraph.

Demonstration. D-only sees only the four observed days, codes pin down the
fever / treatment effect early, the LLM note resolves the late-window
worsening that no other stream captures. The fused posterior reproduces all
three signal sources; the marginal contribution of each can be ablated by
dropping its log-score from the sum.

Run:
    uv run python replications/flowgp/emr_fusion.py --mock-llm   # programmatic notes
    uv run python replications/flowgp/emr_fusion.py --llm        # real LLM notes
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

HERE = os.path.dirname(os.path.abspath(__file__))
N_DAYS = 14
PER_DAY = 4
M = N_DAYS * PER_DAY
T_DAYS = jnp.linspace(0.0, N_DAYS, M)  # day index in [0, 14]
KAPPA, TAU2, OBS_STD = 1.5, 1.5, 0.35  # smoothing in days

# Narrowed to ONLY the late-window incremental qualitative info -- the early
# fever and antibiotic course are already covered by the labs and codes, so a
# note mentioning them as well double-counts (the LLM rewards candidates for
# features other factors already enforce, per HANDOFF.md circularity caveat).
NOTE = (
    "Late-stay nursing note (this report covers ONLY days 11 to 13 of the "
    "admission; earlier days are addressed elsewhere). During days 11, 12, "
    "and 13 the patient had a clear and sustained CLINICAL WORSENING with a "
    "return of low-grade fever. The severity index on those three days is "
    "well above 1.0 (clearly elevated above the patient's baseline of about "
    "0.3). The early shape of the admission is not part of this note."
)


def kern(a, b):
    d = a[:, None] - b[None, :]
    return TAU2 * jnp.exp(-(d**2) / (2.0 * KAPPA**2))


def true_trajectory(t_days):
    """Severity index: early fever (days 1-5), antibiotic cuts it day 6,
    quiet recovery (days 7-10), then a clear secondary worsening days 11-13.

    Constructed so each stream is *load-bearing*: there are several grid
    points above 1.5 in days 1-6 (rewarded by R50.9), the post-treatment
    mean (days 7-10) is well below 0.5 (rewarded by Z51.11), and days 11-13
    have severity ~ 1.4 (rewarded by the clinical note).
    """
    base = 0.3
    natural_fever = 2.5 * jnp.exp(-(((t_days - 3.5) / 2.0) ** 2))
    treatment_cut = 1.0 - jax.nn.sigmoid((t_days - 6.0) / 0.3)
    late_bump = 2.0 * jnp.exp(-(((t_days - 12.5) / 1.2) ** 2))
    return base + natural_fever * treatment_cut + late_bump


# ---------- factor (a): closed-form GP on lab values --------------------------


def gp_posterior_on_labs(obs_days, y):
    """p(f | D) given noisy labs at obs_days."""
    Xo = obs_days
    Koo = kern(Xo, Xo) + OBS_STD**2 * jnp.eye(obs_days.size)
    Kgo = kern(T_DAYS, Xo)
    sol = jnp.linalg.solve(Koo, jnp.eye(obs_days.size))
    m_pred = Kgo @ sol @ y
    K_pred = kern(T_DAYS, T_DAYS) - Kgo @ sol @ Kgo.T
    return m_pred, 0.5 * (K_pred + K_pred.T)


# ---------- factor (b): structured score from ICD/billing codes ---------------


# code-set encoded as deterministic, non-differentiable score on the trajectory.
def code_score(f, fever_thresh=1.5):
    """q(C_codes | f0) for the code set { R50.9 fever, Z51.11 antibiotic day 6 }.

    R50.9  -> max severity in the pre-treatment window (days 1..6) should be
              at least `fever_thresh` = 1.5.
    Z51.11 -> the post-treatment quiet period (days 7..10) should have mean
              severity at most 0.5 (the antibiotic worked).

    Both pieces are non-differentiable (max / threshold lookup) and so qualify
    as evaluate-only constraints: there is no truncated-Gaussian form, no
    forward generator -- only this scalar score per trajectory.
    """
    t = T_DAYS
    pre = (t >= 1.0) & (t <= 6.0)
    post = (t >= 7.0) & (t <= 10.0)
    very_neg = jnp.full_like(f, -1e9)
    fever_max = jnp.max(jnp.where(pre, f, very_neg), axis=-1)
    post_mean = jnp.sum(f * post, axis=-1) / post.sum()
    fever_term = -jnp.maximum(fever_thresh - fever_max, 0.0) ** 2 / 0.3
    treat_term = -jnp.maximum(post_mean - 0.5, 0.0) ** 2 / 0.1
    return fever_term + treat_term


# ---------- factor (c): clinical-note score (LLM or mock) ---------------------


def mock_note_score(f, target_late_mean=1.4):
    """Stand-in for an LLM note verifier: rewards a sustained late-window
    severity > target_late_mean during days 11..13."""
    t = T_DAYS
    late = (t >= 11.0) & (t <= 13.0)
    late_mean = jnp.sum(f * late, axis=-1) / late.sum()
    return -jnp.maximum(target_late_mean - late_mean, 0.0) ** 2 / 0.1


# ---------- driver: pyloop sampler driven by a SUM of log-scores --------------


def fuse_three_factors(
    key,
    m_pred,
    K_pred,
    factor_fns,  # list of callables, each (f0) -> per-candidate log-score
    factor_temps,  # list of floats, divide each factor's log-score
    n_samples=200,
    T=300,
    S=64,
    v_max=50.0,
    t_min=1e-2,
):
    """FlowGP gradient-free guidance with multiple evaluate-only factors.

    The combined log-score per candidate is the per-factor sum (each divided
    by its own temperature). Everything else is identical to the single-factor
    version -- product-of-experts comes out of the softmax for free.
    """
    from flowgp import make_schedule, snr_uniform_grid

    M_ = m_pred.shape[0]
    beta, alpha, log_snr = make_schedule()
    L = jnp.linalg.cholesky(K_pred + 1e-6 * jnp.eye(M_))
    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts = ts[:-1] - ts[1:]
    alphas = alpha(ts[:-1])
    betas = beta(ts[:-1])
    key, k0 = jax.random.split(key)
    f_hat = jax.random.normal(k0, (n_samples, M_))

    try:
        from tqdm import tqdm

        it = tqdm(range(T), desc="EMR-fuse")
    except ImportError:
        it = range(T)

    for i in it:
        a, b, dt = alphas[i], betas[i], dts[i]
        key, ke = jax.random.split(key)
        eps = jax.random.normal(ke, (n_samples, S, M_))
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps
        f0 = f0_hat @ L.T + m_pred
        combined = jnp.zeros((n_samples, S))
        for fn, temp in zip(factor_fns, factor_temps):
            combined = combined + fn(f0) / temp
        w = jax.nn.softmax(combined, axis=1)
        e_c = jnp.einsum("ns,nsm->nm", w, f0_hat)
        v = 0.5 * b * a / (1.0 - a**2 + 1e-6) * (e_c - a * f_hat)
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(nrm / v_max) / (nrm + 1e-8) * v
        f_hat = f_hat + dt * v

    return f_hat @ L.T + m_pred


# ---------- llm wrapper -------------------------------------------------------


def _make_llm_note_fn(args):
    """Build a factor function for the clinical-note LLM score."""
    from llm_score import LLMVerifier, MockVerifier, make_score_fn

    if args.llm:
        backend = LLMVerifier(
            model_name=args.model,
            mode=args.mode,
            x_grid=np.asarray(T_DAYS),
            n_show=args.n_show,
            max_batch=args.batch,
            device=args.device,
        )
    else:
        backend = MockVerifier(mock_note_score)
        backend.x_grid = np.asarray(T_DAYS)
    score_fn = make_score_fn(backend, NOTE, temperature=1.0)
    # `temperature` will be applied inside fuse_three_factors via factor_temps.
    return score_fn, backend


# ---------- main --------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", choices=("yesno", "seqlp"), default="yesno")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-show", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--T", type=int, default=150)
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--temp-codes", type=float, default=1.0)
    parser.add_argument("--temp-text", type=float, default=8.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if not (args.llm or args.mock_llm):
        # default to mock so the script is runnable everywhere
        args.mock_llm = True
    key = jax.random.PRNGKey(0)
    truth = true_trajectory(T_DAYS)

    # D: sparse noisy labs.
    obs_days = jnp.array([1.0, 3.0, 5.0, 7.0])
    key, k1, k2 = jax.random.split(key, 3)
    y_obs = (
        jnp.array([true_trajectory(d) for d in obs_days.tolist()])
        + OBS_STD * jax.random.normal(k1, (obs_days.size,))
    )
    m_pred, K_pred = gp_posterior_on_labs(obs_days, y_obs)

    # D only baseline.
    L = jnp.linalg.cholesky(K_pred + 1e-6 * jnp.eye(M))
    d_only = m_pred + jax.random.normal(k2, (200, M)) @ L.T

    note_fn, backend = _make_llm_note_fn(args)

    # Ablations: D+codes, D+text, D+codes+text
    posteriors = {}
    print("[fuse] D + codes")
    key, k = jax.random.split(key)
    posteriors["D + codes"] = fuse_three_factors(
        k,
        m_pred,
        K_pred,
        [code_score],
        [args.temp_codes],
        n_samples=args.n_samples,
        T=args.T,
        S=args.S,
    )
    print(f"[fuse] D + text ({'LLM' if args.llm else 'mock'})")
    key, k = jax.random.split(key)
    posteriors["D + text"] = fuse_three_factors(
        k,
        m_pred,
        K_pred,
        [note_fn],
        [args.temp_text],
        n_samples=args.n_samples,
        T=args.T,
        S=args.S,
    )
    print("[fuse] D + codes + text  (the killer-app three-factor product)")
    key, k = jax.random.split(key)
    posteriors["D + codes + text"] = fuse_three_factors(
        k,
        m_pred,
        K_pred,
        [code_score, note_fn],
        [args.temp_codes, args.temp_text],
        n_samples=args.n_samples,
        T=args.T,
        S=args.S,
    )
    print(f"  [note-LLM calls: {backend.calls} prompts={backend.total_prompts}]")
    backend.close()

    # report
    def rmse(s, mask):
        return float(
            jnp.sqrt(jnp.mean((s.mean(0)[mask] - truth[mask]) ** 2))
        )

    late_mask = jnp.asarray(T_DAYS) >= 11.0
    early_mask = (jnp.asarray(T_DAYS) >= 1.0) & (jnp.asarray(T_DAYS) <= 6.0)
    middle_mask = (jnp.asarray(T_DAYS) >= 7.0) & (jnp.asarray(T_DAYS) <= 10.0)
    print()
    print(
        f"  {'posterior':22s}  early (1-6)  mid (7-10)  late (11-13)  full"
    )
    for name, s in [("D only", d_only), *posteriors.items()]:
        print(
            f"  {name:22s}  "
            f"{rmse(s, early_mask):.2f}         "
            f"{rmse(s, middle_mask):.2f}        "
            f"{rmse(s, late_mask):.2f}          "
            f"{rmse(s, jnp.ones(M, bool)):.2f}"
        )

    _plot(truth, obs_days, y_obs, d_only, posteriors, args.tag)


def _plot(truth, obs_days, y_obs, d_only, posteriors, tag):
    panels = [("Lab values only  p(f|D)", d_only), *posteriors.items()]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (title, s) in zip(axes, panels):
        lo, hi = jnp.quantile(s, jnp.array([0.05, 0.95]), axis=0)
        ax.fill_between(T_DAYS, lo, hi, color="tab:blue", alpha=0.2, label="5-95%")
        ax.plot(T_DAYS, s.mean(0), color="tab:blue", label="mean")
        ax.plot(T_DAYS, truth, "tab:green", lw=1.5, label="truth")
        ax.scatter(obs_days, y_obs, color="red", s=30, zorder=5, label="labs")
        ax.axvspan(11, 13, color="tab:orange", alpha=0.10, label="late note window")
        ax.axvline(6.0, color="purple", ls="--", lw=1, alpha=0.5, label="antibiotic")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("day of admission")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("severity index")
    fig.suptitle(
        "EMR three-factor fusion: labs (D) + ICD/billing codes + free-text note",
        fontsize=10,
    )
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(HERE, f"emr_fusion{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
