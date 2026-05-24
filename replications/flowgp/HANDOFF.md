# FlowGP replication + investigation — handoff

Branch: **`claude/paper-method-replication-sAOh1`** (repo `flaxter/dl4bi`).
Get it: `git fetch origin && git checkout claude/paper-method-replication-sAOh1`,
then `uv sync --extra cpu` (or `--extra cuda12`). Run anything with
`uv run python replications/flowgp/<script>.py`. Everything is JAX, float64.

## What this is
An independent JAX replication of **FlowGP** ("Conditioning Gaussian Processes on
Almost Anything", arXiv:2605.21041 — GP conditioning recast as a guided
probability-flow ODE) plus a long, skeptical investigation of *where FlowGP is
actually the right tool*.

## Conclusion of the investigation (so you don't re-derive it)
FlowGP is an elegant unification but is **dominated almost everywhere** by a
specialized tool:
- Gaussian prior/posterior → Cholesky, or **`spde_whitening.py`** (exact `O(m)`).
- Linear inequality constraints (monotone/convex/bounded) → these are a
  **truncated Gaussian** (`truncated_gaussian_monotone.py`); exact samplers exist;
  or just a **direct generator** (`direct_monotone_prior.py`).
- Physics (ODE/PDE) → **simulate it** (`direct_sim_baseline.py` beats FlowGP).
- Amortized priors / kernel inference → **DeepRV / neural SBI**
  (`npe_vs_flowgp.py`: NPE matches quality at ~30x fewer simulator runs and
  amortizes).

**FlowGP's one irreducible niche:** conditioning a GP on a constraint that is
*nonlinear, non-generatable, and only evaluable* — i.e. an **LLM / black-box
score** of how well a function matches text. There is no truncated-Gaussian form
and no forward generator; only FlowGP's **gradient-free guidance** (Fisher
identity, paper Eq. 35) applies — it needs *evaluations only*, never gradients.

## The killer application = multimodal fusion
`pi(f0 | D, C) ∝ p(f0 | D) · q(C | f0)`: combine quantitative data `D` (closed-form
Gaussian conditioning) with qualitative/structured `C` (evaluate-only guidance).
Nothing else fuses numbers with free text. Two motivating settings:

1. **Epidemic nowcasting** (`sitrep_fusion.py`): case counts `D` + a WHO sitrep
   `C`. Demo: data stops at the trough (reporting lag) so `D` alone reverts to
   "epidemic over"; the sitrep "renewed increase in the final weeks" recovers the
   true resurgence (late-window RMSE 1.14 → 0.30).
2. **Electronic medical records** (user's idea — not yet built): one latent
   patient trajectory `f0` fused from heterogeneous streams via a
   product-of-experts:
   - quantitative **lab/vital values** → `p(f0|D)` (closed-form Gaussian);
   - **ICD/billing codes** → `q(C_codes|f0)` (a code-consistency score, evaluate-only);
   - free-text **clinical notes** → `q(C_text|f0)` (LLM score, evaluate-only).
   The guidance terms simply **add**, so multiple non-Gaussian factors compose.

## Next task (this is why you're here)
Replace the **programmatic verifier** stand-in with a **real open-weights LLM
likelihood** `q(C | f0)`. The guidance machinery is already done and reusable —
you only swap the score function.

- **Where:** `blackbox_conditioning.py` (replace `describe_logscore`) first, then
  `sitrep_fusion.py` (replace `sitrep_score`). Both call a `score(f0)` that takes
  `f0` of shape `(..., M)` and returns a scalar **log-score per candidate**,
  batched over the `(N, S)` candidates drawn each ODE step. The loop
  (`blackbox_guided_sample` / `fuse`) does not change.
- **How to build `q(C|f0)`:** render the trajectory `f0` as text (e.g. rounded
  values, "week 1: 3, week 2: 7, …"), then either
  (a) **sequence log-prob** — sum the LLM token log-probs of the rendered values
      given a prompt containing the description/sitrep `C` (Biggs & Willis,
      "LLM flow processes for text-conditioned regression"); or
  (b) **verifier logit** — prompt "Is this trajectory consistent with: <C>?
      yes/no" and use the log-prob of "yes".
  (a) is the principled density; (b) is cheaper and more robust. Start with (b).
- **Gradient-free, batched:** only forward LLM calls. Each ODE step scores `S`
  candidates (×`N` samples) — batch them into one LLM call. Use `T≈300`, and
  **larger `S` than the gradient version** (gradient-free needs more; try 64–256).
- **Model:** any local instruct model (Qwen2.5 / Llama-3.x via `transformers`,
  or vLLM/llama.cpp). This is a NEW dependency — add it under a new extra or a
  standalone script; don't bloat the core `dl4bi` deps.

### Gotchas
- **Weight collapse:** if the LLM score is too sharp, the per-step softmax over
  `S` collapses to one candidate. Temper it (divide log-score by a constant) and
  raise `S`. (`simulator_conditioning.py` uses `temp=20`.)
- **Circularity / double-counting:** sitreps and clinical notes are *written from*
  the data, so `q(C|f0)` and `p(f0|D)` share information → the product
  over-concentrates. Apply text only to the *incremental* qualitative content, or
  to data-blind regions (as `sitrep_fusion.py` implicitly does), or down-weight.
- **Validation:** show the fused posterior is *better*, not just narrower — use
  held-out / hindcasting. Easy to fool yourself.
- **Cost:** `T·S` LLM calls per sample — with an LLM that is the whole budget;
  the SPDE/scalability work does **not** help the text half.

## File map
Paper replication: `flowgp.py` (core sampler), `gp_utils.py`,
`monotonic_regression.py` (App. G), `pendulum.py` (H.2), `burgers.py` (H.4).
Investigation: `deeprv_monotonic.py`, `convex_emulator.py`, `pde_emulator.py`,
`direct_sim_baseline.py`, `hmc_monotonic.py`, `fewstep_flowgp_hmc.py`,
`ot_interpolant_flowgp.py`, `direct_monotone_prior.py`,
`truncated_gaussian_monotone.py`, `spde_whitening.py`, `fm_monotonic.py`,
`npe_vs_flowgp.py`.
**The niche (LLM goes here):** `blackbox_conditioning.py`, `simulator_conditioning.py`,
`sitrep_fusion.py`.
(PNGs are git-ignored; each script regenerates its figure on run.)

## Copy-pastable prompt for the new session
> You're picking up a research branch in `flaxter/dl4bi`:
> `claude/paper-method-replication-sAOh1`. Run
> `git fetch origin && git checkout claude/paper-method-replication-sAOh1`, then
> read `replications/flowgp/HANDOFF.md` end-to-end and the docstrings of
> `blackbox_conditioning.py` and `sitrep_fusion.py`.
>
> Context: we replicated FlowGP (arXiv:2605.21041) and established its one
> irreducible use — conditioning a GP on a nonlinear, non-generatable,
> *evaluate-only* likelihood, i.e. an LLM scoring how well a function matches
> text — via gradient-free guidance. A programmatic verifier currently stands in
> for the LLM.
>
> Task: replace that stand-in with a real **open-weights LLM** likelihood
> `q(C|f0)`. I can run a local instruct model. Implement a batched `score(f0)`
> (input `(...,M)`, returns a log-score per candidate) that renders the
> trajectory as text and returns the LLM's log-consistency with a text
> description/sitrep `C` — start with a yes/no verifier logit, then try summed
> token log-probs of the rendered values. Keep it gradient-free (forward calls
> only); reuse the guidance loop in `blackbox_guided_sample`/`fuse` unchanged —
> just swap the score fn and batch the LLM over the `S` candidates per ODE step.
> Do `blackbox_conditioning.py` first (single description), then `sitrep_fusion.py`
> (case data + sitrep). Watch for weight collapse (temper the score, raise `S`)
> and the circularity caveat in the handoff. Then, if it works, sketch the EMR
> version: one latent patient trajectory fused from lab values (Gaussian `D`),
> ICD/billing codes, and free-text notes (each an evaluate-only `q(C|f0)`,
> guidance terms add). Use `uv run python ...`; commit and push to the same branch.
