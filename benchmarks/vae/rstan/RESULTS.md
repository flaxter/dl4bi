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
