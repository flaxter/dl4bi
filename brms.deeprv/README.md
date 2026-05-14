# brms.deeprv

Companion to [`brms`](https://paul-buerkner.github.io/brms/) that injects
**pre-trained deep random-variate decoders** as a `deepRV()` formula
term. Provides spatial / space-time smooths with the same posterior
shape as `brms::gp()` but at a small fraction of the cost at moderate
to large grids.

```r
library(brms.deeprv)

dr <- load_deeprv("unit_interval", grid_size = 100, kernel = "matern_3_2")

fit <- deeprv_brm(
  y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
             ls_prior = prior_uniform(0.05, 0.5)),
  data = df, family = poisson()
)

posterior_predict(fit)
posterior_epred(fit)
plot(conditional_effects(fit))
```

## What it does

A `deepRV()` term replaces the exact Gaussian-process Cholesky of
`brms::gp()` with a forward pass through an MLP trained to emulate
`chol(K(ls)) @ z` for length-scale `ls` drawn from a known range. The
MLP is shipped pre-trained — no Python / JAX / GPU needed at fit time.

At L = 100 the wall-time speedup vs `brms::gp(..., cov = "exp_quad")`
is ~3x; at L = 200 (where `gp()`'s O(L^3) Cholesky bites) it grows
substantially. Both fits give comparable posteriors on the linear
predictor (RMSE ~0.06 at L = 100). Full numbers in
[`benchmarks/vae/rstan/RESULTS.md`](../benchmarks/vae/rstan/RESULTS.md).

## v0.1 feature surface

| Component | Status |
|---|---|
| `load_deeprv(domain, grid_size, kernel)` | 32 shipped decoders: 8 grid sizes × 4 kernels |
| `load_deeprv_kron(...)` | Kronecker composition for 2D `unit_square` |
| `deepRV(s, decoder, obs_idx, ls_prior, ...)` | 1D / Kron spatial term |
| `deepRV_st(...)` | Space-time term with `rw` / `ar1` / decoder time priors |
| `deeprv_brm(formula, data, family, ...)` | Glue: parse, validate, sample via rstan |
| `posterior_predict()`, `posterior_epred()` | S3 methods on `rstantools` generics |
| `conditional_effects()` | Per-grid summary; `plot()` method draws line + CI / 2D heatmap |
| `prior_uniform()`, `prior_exp()` | Prior helpers for `ls`, `sigma_t` |
| Coordinate helpers | `rescale_to_unit_interval()`, `which_grid_points()`, `snap_to_grid()` |
| `by` modes | `by = NULL` (default), `by = numeric` (SVC), `by = factor` (group smooths) |
| `ls_pooling` | `"complete"` / `"none"` / `"partial"` (hierarchical) |
| `gr = TRUE` | Silently accepted; the package's design is already gr-equivalent |
| Families | `poisson()`, `gaussian()` |

## Decoder catalog

Shipped under `inst/extdata/decoders/` (~46 MB total):

| Grid size | Kernels |
|---|---|
| 5, 10, 20, 50, 100, 200, 500, 1000 | matern_1_2, matern_3_2, matern_5_2, rbf |

All trained at 100k steps on a single GPU (~30 min wall). Each .rds
ships its own `(grid_size, kernel, ls_trained_range, weights)`
plus a SHA-256 fingerprint validated at load time.

## Limitations

- **No new-location prediction.** Decoders are trained for a specific
  grid; `posterior_predict(fit, newdata = ...)` errors out. For new
  locations use `brms::gp()` or `snap_to_grid()` first.
- **Priors must fit the decoder's trained range.** Setting an `ls_prior`
  with support outside `dr$ls_trained_range` errors at fit time.
- **Separable kernels only for 2D in v0.1.** Truly isotropic 2D
  Matern arrives in v0.2 as monolithic `unit_square` decoders.

## Install

```r
# From source (not yet on CRAN; the 46 MB catalog exceeds the 5 MB
# tarball limit, so distribution is via GitHub for v0.1):
remotes::install_github("flaxter/dl4bi", subdir = "brms.deeprv")
```

`rstan` is a hard runtime dependency for any fitting; `brms` is
optional but useful for the `posterior_*` generics.

## License

MIT. Authors: Seth Flaxman.

Built on top of the `MLPDeepRV` work in
[`dl4bi`](https://github.com/MLGlobalHealth/dl4bi). See
[`benchmarks/vae/rstan/DESIGN.md`](../benchmarks/vae/rstan/DESIGN.md)
for the design rationale and roadmap.
