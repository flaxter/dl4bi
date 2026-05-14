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
