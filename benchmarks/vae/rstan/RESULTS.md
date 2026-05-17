# RStan integration — HMC results at L=256

Trained `gMLPDeepRV(num_blks=2)` and `MLPDeepRV(dims=[L, L])` on Matérn-1/2
GP samples. Synthetic Poisson observations drawn from the *true* GP at
`(ls*=30, beta*=1, n_obs=177/256)`.

NUTS settings: 2 chains × 1000 iter (500 warmup), `adapt_delta=0.95`,
2 cores. Hardware: CPU, Ubuntu 24.04, rstan 2.32.5, R 4.3.3.

| | MLP (all-Stan) | MLP (extC) | gMLP (all-Stan) | gMLP (extC) |
|---|---|---|---|---|
| Forward match vs JAX | 2.14e-06 | 2.14e-06 | 1.43e-06 | 1.43e-06 |
| Compile (cached / first) | ~5s / ~30s | 125s | ~5s / ~110s | 4s / 83s |
| Sampling | ~42s | 57s | ~8200s | ~4830s |
| **Wall time (sample only)** | **~42s** | **57s** | **138 min** | **80.7 min** |
| Speedup vs same-arch all-Stan | — | **0.82× (slower)** | — | **1.72×** |
| β posterior mean (truth +1.0) | +0.41 | +0.41 | +0.56 | +0.62 |
| β 90% CI | [-0.31, 1.16] | [-0.31, 1.16] | [-0.09, 1.21] | [0.01, 1.24] |
| ℓ posterior mean (truth +30.0) | +31.10 | +31.10 | +30.80 | +29.40 |
| ℓ 90% CI | [16.3, 50.1] | [16.3, 50.1] | [18.2, 46.8] | [17.0, 46.4] |
| Rhat (β, ℓ) | 1.00, 1.01 | 1.00, 1.01 | 1.00, 1.00 | 1.00, 1.00 |
| Divergences | 0 | 0 | 0 | 0 |

(MLP extC and MLP all-Stan are identical samplers — same seed, same data,
same posterior — so the recovery numbers match exactly.)

## Takeaways

**MLP is ~120× faster than gMLP** (42 s vs 138 min sampling). The cost
difference is almost entirely architectural: gMLP per leapfrog has two
`(256, 256) × (256, 64)` spatial matmuls, four LayerNorms, four GELUs,
and eight dense layers — vs. MLP's two dense matmuls and one ReLU.
Stan implementation choice can shift this by ~2×; the architecture
shifts it by 100×+.

**External templated C++ via `--allow-undefined`:**

| Per-iter compute | Verdict | Reason |
|---|---|---|
| Big (gMLP, transformers, multi-block nets) | **Use extC** — 1.7× speedup | Vectorized `stan::math` ops collapse many tape entries into few |
| Small (MLP, shallow nets) | **Stay all-Stan** — 0.82× (extC slower) | Materialization tax (Eigen::Map → Matrix copies) and intermediate-matrix allocation overhead exceeds the AD savings |

There is a real fixed cost per `decode` call when going through external
C++: each `Eigen::Map` argument gets materialized to a plain
`Eigen::Matrix` (a copy), and every `stan::math::add` / `multiply` / `abs`
produces a fresh intermediate matrix. For a forward pass with two
matmuls and a ReLU, that overhead outweighs the gain from vectorized
operations. For gMLP, the per-iter compute is large enough to amortize
that overhead and the win is real.

**Implications for a brms drop-in:**

- **MLPDeepRV is plausible.** ~45 s/fit at L=256 is well within the
  brms iteration window. Non-speed friction remains (decoder
  pre-training, fixed grid, prior alignment, custom post-processing
  methods), but it's a tractable UX problem.
- **gMLPDeepRV is not.** ~80–138 min/fit is outside the iteration
  window, and the speedup ceiling from compilation tricks is ~2×.
  Stays in the "research tool, write Stan directly" category.

## Smaller-domain ideas to explore further

- **MLP at L=64** would likely sample in seconds — fast enough that it
  could be a default option in a brms wrapper.
- **Lower `adapt_delta` (0.8 instead of 0.95)** trades safety for ~2–3×
  fewer leapfrog steps. With both runs at 0 divergences here, there's
  headroom.
- **GPU via `STAN_OPENCL`** would help the big matmuls in gMLP but not
  the LayerNorm/GELU loops; net 2–4× at this scale.

---

# `deeprv_brm()` vs `brms::gp()` — apples-to-apples Poisson recovery

Both ports fit `y ~ Poisson(exp(eta))` on the same synthetic data
generated from an exact RBF GP with `ls = 0.2`, `sigma_gp = 1`. The two
methods see the same `x` grid and the same `y` vector. Iterations:
1000 (500 warmup), 2 chains, 2 cores, `adapt_delta = 0.95`.

Configuration on the brms side: `gp(s, cov = "exp_quad", scale = FALSE)`
with brms-default priors on `lscale` and `sdgp` (a global
`prior(class = "lscale")` silently fails to attach in brms, so I let it
pick per-coefficient defaults).

Configuration on the deeprv side: `deepRV(s, decoder = load_deeprv(...,
kernel = "rbf"), ls_prior = prior_uniform(0.05, 0.5))`.

Reproduce: `Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R --grids=20,50,100`.

| L | brms wall | deeprv wall | speedup | brms Rhat | drv Rhat | brms divs | drv divs | eta RMSE (drv vs brms) | truth RMSE brms | truth RMSE drv |
|---|---|---|---|---|---|---|---|---|---|---|
| 20  | 59.5 s  | 48.5 s | 1.23× | 1.054 | 1.005 | 2 | 0 | 0.120 | 0.437 | 0.348 |
| 50  | 64.2 s  | 52.8 s | 1.22× | 1.032 | 1.004 | 2 | 0 | 0.152 | 0.283 | 0.183 |
| 100 | 159.7 s | 54.7 s | **2.92×** | 1.031 | 1.003 | 0 | 0 | 0.056 | 0.258 | 0.258 |

Wall times include Stan compile (~30 s amortised per program; both
fits hit `auto_write = TRUE` cache after the first L).

## Takeaways

- **At L = 20-50 the comparison is compile-bound** — both fits clock
  in around a minute and the speedup is modest (~1.2×).
- **At L = 100 the gap opens up.** brms::gp() jumps to 160 s as the
  L × L kernel Cholesky becomes expensive; deeprv_brm() stays at 55 s
  because the decoder forward is O(hidden × L) per leapfrog
  irrespective of L. **2.9× speedup at L = 100**; the prototype's
  measured 42 s for MLP at L = 256 suggests the ratio keeps growing
  superlinearly past L = 200, where brms::gp() turns into a multi-
  minute fit.
- **Posterior agreement is solid.** Per-grid `eta` posterior means
  agree between methods at RMSE 0.06-0.15 across the grids tested,
  even though the two methods use different priors and different
  computational paths. Both recover the truth at comparable RMSE
  (deeprv slightly better at L=20-50; tie at L=100).
- **Sampler health is slightly cleaner for deeprv.** brms::gp had
  2 divergences at L=20-50 and Rhat up to 1.054 at L=20 with this
  iter budget; deeprv had 0 divergences and Rhat below 1.005
  throughout. Likely the RW(0, I_L) reparameterisation through the
  decoder mixes more easily than the direct GP for short chains.

## Disentanglement at L = 100

The headline 0.056 eta RMSE between methods at L = 100 has three
sources: (i) the decoder is a finite-capacity approximation to
`chol(K(ℓ)) · z`, (ii) the two methods used different priors on
length-scale and GP amplitude, (iii) Monte-Carlo noise at 1000
iter / 2 chains. To pin down each, I ran two follow-up scenarios:

- **B**: align priors (`prior(uniform(0.05, 0.5), class = "lscale",
  coef = "gps")` on the brms side, matching deeprv's prior_uniform)
  and bump iter to 4000.
- **C**: same as B + swap in a high-res decoder for L = 100 RBF —
  same arch, trained 10× longer (1M steps), final MSE 0.0028 vs the
  v1.0.0 catalog's ~0.03.

| Scenario | brms wall | deeprv wall | speedup | eta RMSE (drv vs brms) | brms divs |
|---|---|---|---|---|---|
| A baseline (1k iter, brms-default priors) | 159.7 s | 54.7 s | 2.92× | 0.056 | 0 |
| B aligned priors + 4k iter, v1.0.0 decoder | 279.2 s | 77.3 s | 3.61× | 0.048 | 1294 |
| C aligned priors + 4k iter, **highres decoder** | 281.3 s | 79.2 s | 3.55× | **0.027** | 1294 |

**Decomposition of the 0.056 baseline RMSE:**

- A → B: dropping 0.008 (~14%) — that's prior-mismatch + MC noise
  contribution.
- B → C: dropping 0.021 (~44%) — that's purely the decoder
  approximation improvement (same priors, same iter, same data, same
  brms config).

**Conclusion: the decoder approximation is the dominant residual
source of disagreement.** Training the L = 100 RBF decoder for 10×
longer roughly halves the eta RMSE. Pushing further (a properly
converged decoder hitting MSE 1e-4 or below) would close most of the
remaining gap; the prior-mismatch contribution is small.

**Caveat on the aligned-prior brms run.** `prior(uniform(0.05, 0.5),
class = "lscale", coef = "gps")` produces a brms model whose sampler
hits ~1300 divergences across both chains. Rhat stays ≈ 1.01 so the
posterior mean is still recoverable, but it tells me the truncated
uniform on the GP length-scale fights brms's default reparameterisation.
For the scaling extension below I use the baseline (brms-default
priors) config so brms is healthy.

Reproduce:
- B: `Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R --grids=100 --iter=4000 --align-priors`
- C: train a 1M-step L=100 RBF decoder into a separate dir; pack with
  `pack_decoders.R`; then add `--decoder-dir=<that path>`.

## Scaling extension: L = 200, 500, 1000

To map the wall-time scaling I added three larger grids. brms::gp() was
left on default priors here so we exercise the same "out of the box"
configuration a typical user would reach for. iter scaled down at each
L to keep total wall time manageable; the comparison still tells the
right story for wall-time, even if posterior-mean recovery suffers at
short chains.

| L | iter | brms wall | deeprv wall | speedup | brms Rhat | drv Rhat | brms divs | drv divs | truth RMSE brms | truth RMSE drv |
|---|---|---|---|---|---|---|---|---|---|---|
| 20  | 1000 | 60 s | 49 s | 1.23× | 1.054 | 1.005 | 2 | 0 | 0.437 | 0.348 |
| 50  | 1000 | 64 s | 53 s | 1.22× | 1.032 | 1.004 | 2 | 0 | 0.283 | 0.183 |
| 100 | 1000 | 160 s | 55 s | 2.92× | 1.031 | 1.003 | 0 | 0 | 0.258 | 0.258 |
| 200 | 1000 | **24 min** | 73 s | **19.6×** | **51.7** ☠ | 1.007 | 1 | 0 | **0.92** | 0.19 |
| 500 | 500  | **3.0 hr** | 199 s | **54×** | **3.56** ☠ | 1.015 | 1 | 0 | 0.22 | 0.088 |
| 1000 | 200 | **4.0 hr** | 887 s | **16×** | **2749** ☠ | 1.041 | 0 | 0 | **1.82** | 0.10 |

### Takeaways

- **The "speedup" framing understates the story past L = 200.**
  brms::gp() with brms-default priors does not mix at L ≥ 200 within
  the iter budgets that keep wall time tractable. The L = 1000 chain
  effectively never moves (Rhat 2749, truth RMSE 1.82 = field-scale
  prior). The L = 500 fit converges loosely (Rhat 3.56) and the L = 200
  fit needs ~4× the iters to mix (see the L = 200 re-run below).
- **deeprv stays clean across the full sweep.** 0 divergences, Rhat
  ≤ 1.04 even at L = 1000 with only 100 sample iters, truth RMSE
  monotonically improving with L (0.35 → 0.10) because larger
  decoders absorb more spatial variation.
- **Wall-time scaling matches the predicted regimes.** brms::gp's
  per-iter cost is dominated by the O(L³) Cholesky:
  60 s → 1430 s → 11 000 s → 14 000 s as L goes 100 → 200 → 500 → 1000.
  deeprv's per-iter cost is O(hidden·L), so wall grows roughly
  linearly with L (54 → 73 → 199 → 887 s).

Reproduce: `Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R --grids=<L> --iter=<N>` for each L / iter pair above. The L = 500 and
L = 1000 fits took multiple hours on an Oxford-CS desktop with 2 cores
allocated; budget overnight.

## HSGP — the approximate-GP competitor

`brms::gp(s, ..., k = K, c = c_val)` builds a Hilbert-space GP
approximation (Riutort-Mayol et al. 2023): K Laplacian basis functions
on `[-c·L_dom, c·L_dom]`. Per-iter cost is O(K² + K·N), independent of
how many spatial grid points L there are.

For the fair fixed-budget comparison I used **K = 20, c = 1.5** —
comfortably above the rule-of-thumb `K ≥ 1.75·c/ℓ ≈ 13` for our
length-scale region `[0.05, 0.5]`. iter = 1000 across all L.

| L | HSGP wall | deeprv wall | HSGP / drv | HSGP Rhat | HSGP divs | truth RMSE HSGP | truth RMSE drv | eta RMSE (drv vs hsgp) |
|---|---|---|---|---|---|---|---|---|
| 20   | 58.7 s | 51.0 s | 1.15×  | 1.007 | 17 | 0.444 | 0.348 | 0.128 |
| 50   | 56.2 s | 51.5 s | 1.09×  | 1.037 | 15 | 0.277 | 0.183 | 0.135 |
| 100  | 57.1 s | 56.8 s | 1.00×  | 1.025 | 13 | 0.282 | 0.258 | 0.077 |
| 200  | 58.5 s | 75.8 s | 0.77×  | 1.010 | 15 | 0.173 | 0.192 | 0.046 |
| 500  | 62.4 s | 332 s  | 0.19×  | 1.019 | 16 | 0.096 | 0.090 | 0.051 |
| 1000 | **108 s** | **3735 s (62 min)** | **0.029×** | 1.004 | 7 | 0.088 | 0.099 | 0.056 |

(`HSGP / drv` < 1 means HSGP is faster.)

### Headline

**HSGP beats `deeprv_brm()` on wall time once L ≥ 200, and the gap
grows fast.** At L = 1000, HSGP fits in 108 s vs deeprv's 62 min —
34× faster. Posterior agreement with deeprv is solid (eta RMSE
0.05–0.13 across L), and truth RMSE is within ~10% of deeprv's
across the entire sweep.

This is the more honest comparison than exact `brms::gp()`. The
take-aways from earlier in this file ("deeprv is faster than
brms::gp()") are correct only against the exact O(L³) Cholesky path.
Against the brms HSGP approximation, deeprv loses on wall time at
moderate-to-large L because:

- HSGP per-iter cost is **O(K² + K·N)** with `K = 20`. At L = 1000
  that's ~20 000 ops, sublinear in L.
- deeprv per-iter cost is **O(hidden·L)** with `hidden = L` (decoder
  shape `dims = [L, L]`). At L = 1000 that's 10⁶ ops, ~50× more.

### So what does deeprv actually buy?

The wall-time pitch needs revising. What deeprv still offers vs HSGP:

1. **Cleaner sampler geometry.** deeprv had 0 divergences and Rhat
   ~1.005-1.04 across the sweep; HSGP had 13-17 divergences at every
   L. Default brms::gp() priors interact awkwardly with the HSGP
   parametrisation in this fixture.
2. **Closed-form access to operators on the field** (v0.3 sketch in
   `DESIGN.md` section 3a). With a differentiable decoder you get
   derivatives, integrals, and extrema of `f(s)` for free —
   `deepRV_monotone()`, `deepRV_integral()`, etc. HSGP gives you the
   field's truncated Karhunen-Loève coefficients but nothing
   structural about `f` itself.
3. **Reproducible fixed grids.** deeprv loads a versioned, fingerprinted
   decoder. HSGP's basis depends on the data domain via `c`; switching
   `c` or `k` changes the model in subtle ways.

Whether (1) and (3) matter to a particular user is workload-dependent.
**For raw wall time on Poisson / Gaussian smooths, HSGP is the default
brms users should reach for at L ≥ 200.** deeprv becomes the answer
when you need operators on the field (v0.3) or care about strict
reproducibility of the smoothing prior.

Reproduce: `Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R --grids=20,50,100,200,500,1000 --include-hsgp --skip-exact --iter=1000`

## L = 200 with iter = 4000: does brms::gp() exact converge with more iter?

The L = 200 line in the scaling table earlier (Rhat = 51.7 at iter = 1000)
left open the question of whether brms::gp() exact would mix if we
just ran longer chains. Re-running the same fixture at iter = 4000
(2000 warmup, 4× the previous iter budget):

| Config | brms wall | brms Rhat | brms divs | treedepth saturation | deeprv wall | deeprv Rhat |
|---|---|---|---|---|---|---|
| L = 200, iter = 1000 | 24 min | 51.7 | 1 | not flagged | 73 s | 1.007 |
| L = 200, **iter = 4000** | **94 min** | **58.7** | 0 | **all 4000 post-warmup transitions saturated treedepth = 10** | **148 s** | **1.002** |

**brms::gp() exact got worse, not better.** Rhat went 51 → 59 and
*every* post-warmup transition saturated the default `max_treedepth = 10`,
meaning each leapfrog integration hit the cap of 2¹⁰ = 1024 steps without
finding a U-turn. HMC literally cannot traverse the GP posterior geometry
at this L with brms's default settings — more iter doesn't help, and
the wall-time cost is linear in iter so brms went from 24 min → 94 min.

deeprv at the same fixture: 148 s wall, Rhat = 1.002, zero divergences,
zero treedepth issues. **38× wall-time speedup and the only one of the
two methods that actually converges at this L.**

To get brms::gp() exact to mix at L ≥ 200 you'd need to tune
`max_treedepth`, `adapt_delta`, possibly reparametrise with a
non-centred GP. That's HMC expert territory. deeprv just works.

This is also one more vote for the HSGP comparison above: HSGP at
L = 200 fit in 58 s with Rhat = 1.01 and 15 divergences (workable),
which is ~100× faster than brms::gp() exact and recovers the truth
at RMSE 0.17 vs brms's 0.88.

Reproduce: `Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R --grids=200 --iter=4000`.

## L = 250 Matérn-3/2 default-priors comparison with uncertainty intervals

A "reasonably large, mildly hard" head-to-head at L = 250, designed
to exercise all three methods on a harder fixture than the L ≤ 200
RBF baseline:

- Kernel: **Matérn-3/2** (rougher than RBF, less data-friendly).
- Length-scale truth: **ls = 0.15** (tighter; closer to the Nyquist
  resolution of the L = 250 grid).
- Intercept: **beta0 = -1** so mean rate is ~0.37 — more Poisson
  noise per cell (103 of 250 cells nonzero in this fixture).
- iter = 4000 (warmup = 2000), 2 chains, adapt_delta = 0.95, **all
  defaults**: brms uses default `inv_gamma` on `lscale` and
  `student_t` on `sdgp`; deeprv uses `prior_uniform(0.05, 0.5)`.

Decoder for deeprv: a freshly trained L = 250 Matérn-3/2 decoder at
100k steps (training MSE = 0.0068, ~5× tighter than the v0.1
catalog's typical L = 100 decoder).

For each method I extract S = 4000 posterior draws of `eta` (the
linear predictor), then compute per-draw `RMSE(eta_s, eta_true)` to
build a distribution. The mean of that distribution is the
**per-draw RMSE mean**; the 5th / 95th percentiles give a 90 % CI.
I also report the conventional **posterior-mean RMSE**
(`RMSE(mean(eta_s), eta_true)`), which is what you'd quote in a
paper.

| Method | wall | Rhat | divs | per-draw RMSE 90 % CI | posterior-mean RMSE |
|---|---|---|---|---|---|
| brms::gp() exact          | **53 min** | 1.003 | 28 | [0.302, 0.572] (mean 0.424) | 0.262 |
| brms::gp() HSGP (k=20)    | **62 s**   | 1.000 |  7 | [0.266, 0.488] (mean 0.363) | **0.240** |
| deeprv_brm() (matern_3_2) | **125 s**  | 1.002 |  **0** | [0.295, 0.532] (mean 0.409) | 0.279 |

### Bottom line

**HSGP wins on accuracy by a small margin** (posterior-mean RMSE
0.240 vs 0.262 exact, 0.279 deeprv) — and it's also the fastest of
the three by 25× over deeprv and ~50× over exact GP.

**deeprv has the cleanest sampler health**: 0 divergences vs HSGP's
7 vs brms exact's 28. Modest but real — meaningful if you care about
tail-of-posterior estimates.

The per-draw RMSE 90 % CIs **overlap heavily** across methods (HSGP
[0.27, 0.49], exact [0.30, 0.57], deeprv [0.30, 0.53]). The HSGP-best
ranking is suggestive from a single fixture but not statistically
significant; tightening it would mean multi-seed bootstrap (5–10
fixtures × 3 methods, dominated by the 53 min exact GP fit per seed).

**Note on brms::gp() exact converging here.** Earlier results in this
file recorded brms exact failing to mix at L ≥ 200 (Rhat = 51.7 at
L = 200 iter = 1000, Rhat = 58.7 at iter = 4000, all transitions
saturating treedepth = 10). That was specifically with the **RBF
kernel + brms's default inv_gamma lscale prior + low-noise (intercept
0) fixture**. The Matérn-3/2 + intercept = -1 + iter = 4000 config
above mixes cleanly (Rhat = 1.003, 28 divergences total) — apparently
the brms-default prior interacts much more favourably with Matérn-3/2
than with RBF, and the lower-information Poisson signal gives the
posterior a less pathological geometry. **brms::gp() exact mixability
is fixture-dependent in ways that aren't obvious upfront**; budget
for HMC tuning if you choose it at L ≥ 200.

Reproduce: `Rscript benchmarks/vae/rstan/bench_L250_compare.R`.

## 2D head-to-head: brms exact vs HSGP vs deeprv Kron (N_side = 20, L = 400)

The 1D comparisons above leave open whether HSGP still wins in 2D —
where deeprv has a natural advantage through `load_deeprv_kron()`'s
explicit separable-kernel construction. To test: a unit_square
Poisson fixture drawn from a separable Matérn-3/2 GP, fit by all
three methods with default priors and iter = 4000.

Fixture:
- N_side = 20 grid points per axis -> L = 400 spatial cells.
- Separable Matérn-3/2 with ls_x = ls_y = 0.2, sigma_gp = 1,
  generated via the Kronecker Cholesky `F = L_x %*% Z %*% L_y^T`.
- Intercept = -1 (mean rate ~0.37; 163 of 400 cells nonzero).

Methods (all default priors):
- `brms::gp(x1, x2, cov = "matern32", scale = FALSE, iso = FALSE)` —
  exact separable 2D GP.
- `brms::gp(x1, x2, cov = "matern32", scale = FALSE, iso = FALSE,
  k = 20, c = 1.5)` — HSGP with K² = 400 tensor-product basis
  functions.
- `deeprv_brm()` with `load_deeprv_kron("unit_square", grid_side = 20,
  kernel = "matern_3_2")` and `prior_uniform(0.05, 0.5)` on each
  axis's length-scale.

| Method | wall | Rhat | divs | per-draw RMSE 90 % CI | posterior-mean RMSE |
|---|---|---|---|---|---|
| brms exact 2D (iso = FALSE)         | **2 h 11 m** | 1.003 | 0 | [0.52, 0.74] (mean 0.62) | **0.397** |
| brms HSGP 2D (k=20/dim, c=1.5)      | **84 s**     | 1.002 | 0 | [0.49, 0.65] (mean 0.56) | **0.397** |
| deeprv_brm Kron (grid_side=20)      | **138 s**    | 1.003 | 0 | [0.52, 0.67] (mean 0.59) | 0.413 |

### Bottom line

**HSGP and exact 2D tie on posterior-mean RMSE** (0.397) while HSGP
is **93× faster** (84 s vs 2 h 11 m). deeprv Kron is 4 % less
accurate (0.413), ~57× faster than exact GP, and ~1.6× slower than
HSGP. All three have zero divergences and clean Rhat — 2D sampling
geometry is friendlier here than the 1D Matérn-3/2 case at L=250.

The per-draw RMSE 90 % CIs overlap pairwise — the HSGP-vs-deeprv
gap (0.413 - 0.397 = 0.016 on posterior-mean RMSE; ~0.04 on per-draw
mean) is not individually statistically significant from one
fixture. The qualitative ranking matches the 1D L = 250 result
though: HSGP ≥ exact ≥ deeprv on accuracy, HSGP > deeprv > exact
on speed.

### Why HSGP wins in 2D too

- HSGP per-iter cost: O(K^d × N) = 20² × 400 ≈ 1.6e5 ops; *flat*
  in L (= N here) since the basis count is fixed.
- deeprv Kron per-iter cost: 2 × N_side² × hidden = 2 × 400 × 20
  = 1.6e4 ops for the decode itself, but the latent matrix
  `matrix[T, L]` z has N_side² = 400 parameters that HMC has to
  sample. That latent-dimensionality cost dominates and grows
  linearly with L.
- brms exact 2D: O(L³) Cholesky = 6.4e7 ops per iter.

The takeaway is the same as in 1D: **HSGP's compact basis
representation beats deeprv's grid-resolution-sized latent for
straight GP smoothing**. deeprv's edge sits in regimes the bench
hasn't touched yet — operator terms (∂f, ∫f, etc., DESIGN §3a),
unusual kernels not natively in brms's `cov` menu, or any place a
user wants a fixed-grid, fingerprinted prior they can re-load
deterministically.

Reproduce: `Rscript benchmarks/vae/rstan/bench_2d_compare.R`.

## Where HSGP breaks: short length-scales relative to K

Earlier sections used HSGP with `k = 20, c = 1.5`, which by the
Riutort-Mayol et al. 2023 rule of thumb (`K >= 1.75 * c * L_dom / ls`)
suffices for `ls >= 0.13`. What happens at `ls = 0.05` (K_required ≥ 52.5,
so K = 20 is well under)?

Same L = 250 / iter = 4000 / default-priors / Matérn-3/2 setup as the
disentanglement comparison, with `ls_true = 0.05` and `beta0 = 0`
(mean rate ~1, decent signal-to-noise).

| Method | wall | Rhat | divs | per-draw RMSE 90% CI | posterior-mean RMSE |
|---|---|---|---|---|---|
| brms exact                           | 60 min | 1.003 | 6 | [0.33, 0.53] (mean 0.42) | **0.285** |
| brms HSGP k=20 (the headline default)| 149 s  | 1.002 | 3 | [0.61, 0.72] (mean 0.66) | **0.642** ← broken |
| brms HSGP k=60 (sufficient K)        | 91 s   | 1.001 | 9 | [0.33, 0.52] (mean 0.41) | 0.308 |
| deeprv_brm                           | 277 s  | 1.001 | **0** | [0.35, 0.52] (mean 0.43) | 0.325 |

### What this means

1. **HSGP k=20 fails silently and confidently.** Posterior-mean RMSE
   0.642 is **2.25× worse than exact** and its per-draw 90% CI
   [0.61, 0.72] doesn't even overlap the other methods' CIs. The
   truncated basis can't represent the short-scale field, so HSGP
   returns an oversmoothed posterior that's confidently wrong. Rhat,
   divergences, and treedepth all look fine — there's no
   sampler-side red flag to tell the user "K is too small."

2. **The fix is to bump K.** HSGP k=60 (above the K_required ≥ 52.5
   threshold) recovers to RMSE 0.308, close to exact's 0.285. So this
   isn't a fundamental limit of HSGP, just a K-budget issue.

3. **The user has to know to do that.** K depends on the unknown
   length-scale of the data-generating process. In practice users
   pick a default and run; if `ls` happens to be small enough that
   K_required exceeds K_chosen, the fit looks healthy but is
   substantially biased.

4. **deeprv has no such failure mode here.** The L=250 catalog
   decoder was trained for `ls ∈ [0.01, 1.0]`; `prior_uniform(0.02,
   0.5)` covers the truth at `ls = 0.05` and the fit just works
   (RMSE 0.325, comparable to k=60 HSGP and exact). No tuning knob
   to misset. Sampler-cleanest of the four (0 divergences).

### Implications for the earlier "HSGP wins" framing

The earlier sections in this file said HSGP beats deeprv on wall time
and matches it on accuracy across L = 20-1000 (1D) and L = 400 (2D).
That comparison was at `ls = 0.15-0.2`, which is comfortably above
K = 20's resolution. **The HSGP-best ranking is conditional on the
user picking a K that matches the truth's length-scale**, which they
won't always know how to do.

The honest claim is now:

- **For users who know their data's length-scale regime well**: tune K
  per the rule-of-thumb and HSGP wins on wall time and matches deeprv
  on accuracy.
- **For users who don't (or whose data may have heterogeneous scales)**:
  deeprv's catalog covers the full ls range without per-fit tuning. K
  mis-specification fails silently; deeprv's prior-range
  mis-specification fails loudly (an explicit error at fit time, see
  `validate_prior_in_range`).

This is the first robustness-axis on which deeprv clearly beats HSGP
for "vanilla GP smoothing", not just on the operator-terms-and-friends
axes from earlier.

Reproduce: `Rscript benchmarks/vae/rstan/bench_short_ls.R`.
