# RStan integration — HMC results at L=256

Trained `gMLPDeepRV(num_blks=2)` on Matérn-1/2 GP samples (10k steps,
final MSE ~0.07). Synthetic Poisson observations drawn from the *true*
GP at `(ls*=30, beta*=1, n_obs=177/256)`.

NUTS settings: 2 chains × 1000 iter (500 warmup), `adapt_delta=0.95`,
2 cores. Hardware: CPU, Ubuntu 24.04, rstan 2.32.5, R 4.3.3.

| | MLPDeepRV (all-Stan) | gMLPDeepRV (all-Stan) | gMLPDeepRV (external C++) |
|---|---|---|---|
| Forward match vs JAX | 2.14e-06 | 1.43e-06 | 1.43e-06 |
| Wall time (compile + sample) | ~4 min | 138.7 min | 80.7 min |
| Speedup vs all-Stan gMLP | — | 1.00× | **1.72×** |
| β posterior mean (truth +1.0) | +0.41 | +0.56 | +0.62 |
| β 90% CI | [-0.31, 1.16] | [-0.09, 1.21] | [0.01, 1.24] |
| ℓ posterior mean (truth +30.0) | +31.10 | +30.80 | +29.40 |
| ℓ 90% CI | [16.3, 50.1] | [18.2, 46.8] | [17.0, 46.4] |
| n_eff (β, ℓ) | 636, 404 | 545, 373 | — |
| Rhat (β, ℓ) | 1.00, 1.01 | 1.00, 1.00 | 1.00, 1.00 |
| Divergences | 0 | 0 | 0 |

**External-C++ takeaway.** The big matmuls (especially the 256×256 SGU
matmul) already used `stan::math::multiply`'s optimized matrix-AD path in
both versions. The wins from external C++ here come from collapsing
small element-wise ops (`gelu_m`, parts of `layer_norm`) onto vectorized
`stan::math` overloads, which trims the tape but doesn't change the
dominant cost. Architectures with many small ops and few big matmuls
would see bigger gains; this one is matmul-bound, so the speedup is
capped at ~2×.
