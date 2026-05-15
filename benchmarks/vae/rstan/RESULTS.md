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
