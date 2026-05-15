# brms.deeprv 0.1.0 (beta release)

Beta tag: `v0.1.0-beta`. Install:

```r
remotes::install_github("flaxter/dl4bi", ref = "v0.1.0-beta",
                        subdir = "brms.deeprv")
```

The beta ships only the **L ≤ 200** decoders (24 of the 32 trained for
v0.1) to keep the installed package under 2 MB. Large-L decoders (500,
1000) are available on the `claude/deeprv-rstan-integration-VDtiw`
branch for power users who need them. The HSGP comparison in
`RESULTS.md` shows that brms's `gp(s, k = 20)` is faster than
deeprv_brm() at L ≥ 200 anyway, so the trimmed catalog covers the
regime where deeprv is the right tool.

First release. Implements DESIGN.md sections 2.1-2.9 plus most of 2.10:

## Decoder catalog

- 32 pre-trained MLPDeepRV decoders shipped under
  `inst/extdata/decoders/` covering grid sizes
  `{5, 10, 20, 50, 100, 200, 500, 1000}` x kernels
  `{matern_1_2, matern_3_2, matern_5_2, rbf}` on the unit interval.
- Each decoder ships with `(arch_version, grid_size, kernel,
  ls_trained_range, weights, JAX golden test cases, SHA-256
  fingerprint)`. Fingerprints are recomputed and validated at load.
- Forward parity (JAX vs rstan via `expose_stan_functions`) verified
  for every shipped decoder at install time, max abs diff < 1e-5.

## API

- `load_deeprv()`, `load_deeprv_kron()` to fetch a decoder from the
  catalog.
- `deepRV()` formula term plus `deepRV_st()` for space-time.
  `by = numeric` (SVC) and `by = factor` (group smooths) supported on
  the 1D `deepRV()` term; `ls_pooling` in
  `{"complete", "none", "partial"}` for `by = factor`.
- `prior_uniform()`, `prior_exp()` prior helpers.
- `deeprv_brm()` wrapper. Family support: `poisson()`, `gaussian()`.
  RHS covariates handled via `model.matrix()`.
- `deepRV_st()`'s `decoder_time` accepts `"rw"`, `"ar1"`, or a 1D
  `deepRV_decoder`. Spatial side accepts 1D or Kronecker decoders.
- `posterior_predict()` / `posterior_epred()` S3 methods registered
  against `rstantools` generics. `conditional_effects()` returns
  per-grid posterior summaries; `plot()` method draws a line + CI
  ribbon for 1D smooths, a viridis heatmap with white contours for
  Kronecker 2D smooths, faceted panels for `by = factor`.
- Coordinate helpers: `rescale_to_unit_interval()`,
  `rescale_to_unit_square()`, `which_grid_points()`, `snap_to_grid()`.

## Benchmarks

Wall time vs `brms::gp(..., cov = "exp_quad")` on the same Poisson
fixture at the same `x` grid (1000 iter, 2 chains, `adapt_delta = 0.95`,
brms-default priors on `lscale` / `sdgp`):

| L | brms::gp() wall | deeprv_brm() wall | speedup |
|---|---|---|---|
| 20 | 59.5 s | 48.5 s | 1.23x |
| 50 | 64.2 s | 52.8 s | 1.22x |
| 100 | 159.7 s | 54.7 s | 2.92x |

At L = 200+ the gap opens further as brms's O(L^3) Cholesky bites.
Posterior `eta` agreement between methods: RMSE 0.056 at L=100,
dominated by decoder approximation error (training MSE 0.03 at L=100;
training the decoder 10x longer halves the RMSE).

## Known limitations

- The 46 MB catalog exceeds CRAN's 5 MB tarball cap, so v0.1 ships
  via GitHub. v0.2 will move large decoders to download-on-demand.
- No new-location prediction; `posterior_predict(fit, newdata = ...)`
  errors with a pointer to `brms::gp()` for that use case.
- True isotropic 2D Matern (rather than axis-wise separable) deferred
  to v0.2.
- `by` / `gr` combined with `deepRV_st` deferred to v0.2.
