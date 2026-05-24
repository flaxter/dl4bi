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

## Status of the LLM swap (May 2026)
The LLM verifier is **done**: `llm_score.py` (`LLMVerifier` + `MockVerifier`
+ `flowgp_pyloop` driver), with `--llm` / `--mock-llm` flags on
`blackbox_conditioning.py`, `sitrep_fusion.py`, and the new `emr_fusion.py`.
Default model is **Qwen 2.5-7B-Instruct** in yesno mode (`log P(Yes) - log P(No)`
to "Is this trajectory consistent with: <C>? Yes / No"); seqlp mode is also
implemented but base-rate dominated for repetitive numerical text — needs a
ratio against an empty-description prefix to be useful.

Results from real-LLM runs (Qwen 7B, RTX 5000 Ada / RTX 4090, 16-24 min/run):

| Demo                              | Metric                           | Programmatic | D-only | LLM        |
| :---                              | :---                             |         ---: |   ---: | ---:       |
| blackbox `(N=16, T=120, S=24, t=4)`| peak loc (target 0.7)            |     0.65±0.08|      — | **0.53±0.21** |
|                                   | peak val (target 2.0)            |     1.50±0.49|      — | **1.73±0.85** |
|                                   | unimodal frac                    |          0.95|      — | 0.88       |
| sitrep   `(N=16, T=100, S=24, t=8)`| late-window RMSE                 |          0.30|   1.14 | **0.85**   |
|                                   | mean late level (truth 1.5)      |          1.50|  −0.01 | 0.39       |
|                                   | P(resurgence: late > 0.8)        |          0.95|   0.22 | 0.38       |
| emr      `(N=16, T=100, S=24, tc=1, tt=4)`| late-window RMSE        |          0.91|   1.81 | D+text **1.24** / D+codes+text 1.82 |
|                                   | full-grid RMSE                   |          0.54|   0.89 | D+text **0.63** / D+codes+text 0.93 |

### What was discovered along the way
- **Circularity dominates broad descriptions.** First sitrep run used a SITREP
  that mentioned BOTH the early peak (already pinned by `D`) and the late
  resurgence; this DEGRADED the late posterior (RMSE 1.14 → 1.52, mean late
  −0.01 → −0.37). Narrowing the description to *only* the late-window
  incremental info turned the result around (RMSE 1.52 → 0.85). The same
  pattern repeated in EMR (broad NOTE: D+text full RMSE 0.55 with
  temp_text=6; narrow NOTE: 0.63 with temp_text=4 — still better than D-only
  0.89, and more honest about the source of the signal).
- **JAX preallocation blocks the LM.** JAX takes 75 % of the card by default;
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MEM_FRACTION=0.2` are now set in
  each `--llm`-capable script.
- **lm_head logits OOM.** Pull the last-position slice BEFORE the float upcast
  (yesno); chunk over the sequence dim (seqlp). With Qwen 7B and prompts ~350
  tokens on bf16, batch up to 48 fits in 32 GB; KV cache, not weights, is the
  spike.
- **Three-factor EMR temp balance is open.** With raw `code_score` in
  [−15, 0] and LLM text score spread ~7–10 logits, the D+codes+text combo
  is still dominated by codes at `temp_codes=1, temp_text=4`; D+text alone
  beats D+codes+text. Worth sweeping `temp_codes ∈ {3,5,8}` and/or
  `temp_text ∈ {1.5,2}` next.

## What still wants doing
1. **Sweep the EMR temperatures** so D+codes+text actually beats D+text alone
   (this is the killer-app punchline; mock smoke test achieves it at full
   RMSE 0.46).
2. **seqlp-ratio mode.** Sequence-log-prob currently rewards predictability of
   repeated tokens (flat-zero trajectories score best regardless of
   description). Subtracting the empty-prefix log-prob would isolate the
   *conditional* information. Useful for the principled-density variant.
3. **Diagnostic: per-step softmax entropy.** Print mean entropy of the
   per-candidate softmax — too low ⇒ weight collapse, too high ⇒ ineffective
   guidance. Currently you tune by eyeball.
4. **Bigger LLM.** Qwen 7B is the sweet spot on a 32 GB card; a 14 B model
   would likely sharpen the calibration but might not fit. Try Qwen 2.5-14 B
   on clpc122's 4090 (24 GB) using 4-bit quant.

## Original next-task brief (for context — superseded by the section above)
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
**The niche (LLM lives here):** `blackbox_conditioning.py`,
`simulator_conditioning.py`, `sitrep_fusion.py`, `emr_fusion.py`.
**LLM machinery:** `llm_score.py` (`LLMVerifier`, `MockVerifier`,
`flowgp_pyloop`, `make_score_fn`).
(PNGs are git-ignored; each script regenerates its figure on run.)

## Copy-pastable prompt for the new session
> You're picking up `flaxter/dl4bi` on branch
> `claude/paper-method-replication-sAOh1`. Run
> `git fetch origin && git checkout claude/paper-method-replication-sAOh1`,
> then read `replications/flowgp/HANDOFF.md` end-to-end and the docstring of
> `replications/flowgp/llm_score.py`.
>
> Context. FlowGP (arXiv:2605.21041) is replicated; its irreducible niche is
> conditioning a GP on a nonlinear non-generatable evaluate-only `q(C|f0)`
> via gradient-free guidance. The verifier is now a real open-weights LLM
> (Qwen 2.5-7B-Instruct, yesno mode). Three demos call it: `blackbox_conditioning.py`
> (single description), `sitrep_fusion.py` (closed-form D + sitrep), and
> `emr_fusion.py` (three-factor product: labs D + ICD/billing codes + clinical
> note). All work; full results table is in HANDOFF.md.
>
> Open work items (in priority order):
> 1. **Sweep EMR temperatures.** The three-factor combo D+codes+text currently
>    underperforms D+text alone with `temp_codes=1, temp_text=4` because
>    codes dominate the additive log-score. Try `temp_codes ∈ {3,5,8}` and/or
>    `temp_text ∈ {1.5,2}` until D+codes+text beats D+text on late-window RMSE.
> 2. **Add a softmax-entropy diagnostic** to `flowgp_pyloop` (mean entropy of
>    the per-candidate softmax per ODE step) so temperature sweeps don't need
>    to be tuned by eyeball.
> 3. **`seqlp_ratio` mode** in `LLMVerifier`: subtract the empty-prefix
>    sequence log-prob so the principled-density score isn't dominated by
>    intrinsic token predictability (flat-zero candidates currently win).
> 4. **Bigger model.** Qwen 14B with 4-bit quant on clpc122 (24 GB 4090) is
>    the next jump up from 7B.
>
> GPU hosts (per CLAUDE memory): clpc35 = RTX 5000 Ada 32 GB, clpc122 = 4090
> 24 GB. JAX preallocation is disabled in the demos so torch has room. Use
> `uv run python ...`. Reproduce any of the three Qwen 7B runs with the exact
> flags in the HANDOFF results table. Commit and push to the same branch.
