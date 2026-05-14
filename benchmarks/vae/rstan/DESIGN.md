# brms.deeprv — Design Document

**Status:** Pre-implementation. The RStan + DeepRV bridge that this package
sits on top of is built and verified on the `claude/deeprv-rstan-integration-VDtiw`
branch of `MLGlobalHealth/dl4bi`. The R-package work has not started.

**Audience:** A fresh agent picking up implementation. Assume you have
**not** read the conversation that produced this doc — everything you need
should be here, with pointers into the prototype code on the branch.

---

## 1 · Context

### 1.1 What's already on the branch

| Path | Role |
|---|---|
| `dl4bi/vae/deep_rv.py` | Original Flax `MLPDeepRV`, `gMLPDeepRV`, `KernelBiasTransformerDeepRV` implementations |
| `benchmarks/vae/deep_rv_example.py` | The canonical NumPyro inference model the RStan port mirrors (Poisson Matérn-1/2, `ls ~ U(1,100)`, `beta ~ N(0,1)`, `z ~ N(0,I_L)`) |
| `benchmarks/vae/rstan/export_mlp_decoder.py` | Train `MLPDeepRV(dims=[L,L])` on Matérn-1/2 GP samples and dump weights + golden test cases + a synthetic Poisson inference fixture to JSON |
| `benchmarks/vae/rstan/mlp_decode.stan` | Stan port of `MLPDeepRV.decode` |
| `benchmarks/vae/rstan/verify_numpy.py` | Pure-numpy mirror of the Stan forward — sanity check independent of R |
| `benchmarks/vae/rstan/check_match.R` | rstan `expose_stan_functions` confirms forward matches JAX to 2e-6 |
| `benchmarks/vae/rstan/run_hmc.R` | Full HMC on the MLP model. **42 s sampling at L=256** |
| `benchmarks/vae/rstan/gmlp_decode.stan`, `export_gmlp_decoder.py`, `verify_gmlp_numpy.py`, `check_gmlp_match.R`, `run_gmlp_hmc.R` | Same pipeline for `gMLPDeepRV(num_blks=2)`. **138 min sampling at L=256.** Forward matches JAX to 1.4e-6 |
| `benchmarks/vae/rstan/gmlp_decode_impl.hpp`, `gmlp_decode_extc.stan`, `run_gmlp_hmc_extc.R` | External templated C++ implementation of gMLP `decode` linked via rstan `--allow-undefined`. **81 min sampling at L=256.** |
| `benchmarks/vae/rstan/RESULTS.md` | Summary table of all four runs (MLP all-Stan, MLP extC, gMLP all-Stan, gMLP extC) |
| `benchmarks/vae/rstan/train_decoders.py` | **§8 step 1.** YAML-driven catalog trainer. Trains one `MLPDeepRV(dims=[L,L])` per `(grid_size, kernel)` on `[0, 1]`, emits a JSON per decoder + a `manifest.json`. Fingerprint-aware skip on rerun. |
| `benchmarks/vae/rstan/pack_decoders.R` | **§8 step 1.** Converts the JSON intermediates into canonical `.rds` files per §2.2 + a `manifest.json` mapping fingerprint → filename. Re-computes the fingerprint in R and aborts if it disagrees with Python. |
| `benchmarks/vae/rstan/verify_rds.R` | **§8 step 1.** Loads a packed `.rds`, exposes `mlp_decode.stan::decode` via rstan, and verifies the embedded JAX test cases match Stan's forward to < 1e-5. The install-time check from §4.8. |
| `benchmarks/vae/rstan/configs/v0.1.yaml`, `configs/smoke.yaml` | Catalog spec for v0.1 (8 grids × 4 kernels, 100k steps) and a 2-decoder smoke spec (~3 s end-to-end). |

### 1.2 Empirical baseline that drove this design

| Decoder | Implementation | Sampling wall time (L=256, 2×1000 NUTS) |
|---|---|---|
| MLPDeepRV | all-Stan | **42 s** |
| MLPDeepRV | external templated C++ | 57 s (slower — see §6) |
| gMLPDeepRV | all-Stan | 138 min |
| gMLPDeepRV | external templated C++ | 81 min |

**The MLP all-Stan number is the foundation of this package.** ~45 s/fit
at L=256 puts a brms-style spatial term inside the iteration window of
typical applied workflows. gMLP at 80–138 min is outside that window and
is explicitly **out of scope for v0.1**.

### 1.3 Design principles

1. **No magic.** No matchers, no on-demand trainers, no silent defaults.
   The user specifies the domain, grid, kernel, and prior explicitly.
2. **CPU-only audience.** Everything ships pre-trained. No reticulate
   dependency, no requirement for users to have JAX/GPU installed.
3. **Grid data only.** This is the price of pre-training. Out-of-sample
   spatial prediction is impossible with `MLPDeepRV`; the package
   refuses irregular geometries and points to `brms::gp()`.
4. **Compose with brms.** Most brms features should work because the
   `deepRV` term contributes additively to `eta`. The only features
   that change the *shape* of the term are `by`, `gr`, `ls_pooling`,
   and distributional models.

---

## 2 · v0.1 — Detailed scope

### 2.1 Decoder catalog (the "library")

**unit_interval (1D, on [0, 1]):**

| Grid size | L | Kernels | Decoders |
|---|---|---|---|
| 5 | 5 | matern_1_2, matern_3_2, matern_5_2, rbf | 4 |
| 10 | 10 | (same 4) | 4 |
| 20 | 20 | | 4 |
| 50 | 50 | | 4 |
| 100 | 100 | | 4 |
| 200 | 200 | | 4 |
| 500 | 500 | | 4 |
| 1000 | 1000 | | 4 |

**8 grid sizes × 4 kernels = 32 unit_interval decoders.**

**unit_square (2D, on [0, 1]²): via Kronecker only in v0.1.** No
monolithic 2D decoders shipped (v0.2 candidate). All 2D spatial models
compose two `unit_interval` decoders along x and y axes, requiring the
kernel to be axis-wise separable. See §2.5.

### 2.2 Decoder artifact format

Each decoder is a single binary `.rds` file shipped inside the package
(`inst/extdata/decoders/`) or downloaded on demand. The R object is a
list with fixed structure:

```r
list(
  schema_version = 1L,
  arch           = "MLPDeepRV",
  arch_version   = "1.0.0",     # bump if the Stan port changes
  domain         = "unit_interval",
  grid_size      = 50L,
  L              = 50L,
  grid_coords    = <numeric vector of length L>,   # the exact training grid
  kernel         = "matern_3_2",
  conditionals   = c("ls"),
  ls_trained_range = c(0.01, 1.0),
  weights = list(
    W1 = <matrix>,  b1 = <numeric vector>,
    W2 = <matrix>,  b2 = <numeric vector>
  ),
  training = list(
    steps = 100000L, batch = 32L, lr = 1e-3,
    final_mse = 0.04,
    seed = 0L
  ),
  fingerprint = "<sha256 of arch + arch_version + domain + grid + kernel + ls_range>"
)
```

The fingerprint is the lookup key. `load_deeprv(...)` errors if a fingerprint
isn't in the shipped manifest.

**Canonical fingerprint format** (implemented in `train_decoders.py` and
`pack_decoders.R`):

```
sha256("<arch>|<arch_version>|<domain>|<grid_size>|<kernel>|<ls_lo>|<ls_hi>")
```

Floats are formatted with `%.8f` (e.g. `0.01000000`, `1.00000000`). This
is the one format that produces byte-identical strings in Python and R —
`repr(float)`, `str()`, `format()`, JSON, etc. all disagree at the edges.
See §4.8.

### 2.3 R-side public API

#### 2.3.1 Loading a decoder

```r
dr <- load_deeprv(
  domain    = "unit_interval",      # required: "unit_interval" or "unit_square"
  grid_size = 100,                  # required: one of the cataloged sizes
  kernel    = "matern_3_2",         # required: matern_1_2, matern_3_2, matern_5_2, rbf
  by        = NULL                  # NULL for plain spatial; see §2.4.3
)
# class(dr): c("deepRV_decoder")
```

For unit_square (Kronecker composition):

```r
dr_2d <- load_deeprv_kron(
  domain      = "unit_square",
  grid_side   = 50,                 # same N for both x and y axes
  kernel      = "matern_3_2"        # axis-wise; will error on non-separable kernels
)
# class(dr_2d): c("deepRV_decoder_kron", "deepRV_decoder")
# Internally loads two unit_interval decoders (one for x, one for y).
```

`print.deepRV_decoder` shows what the decoder expects. `dr$grid_coords`
exposes the exact training locations so users can match their data.

#### 2.3.2 Coordinate frame helpers

These are **explicit** — the user calls them, the package never auto-rescales.

```r
# Linear rescaling to the unit cube.
s_norm <- rescale_to_unit_interval(s, from = c(xmin, xmax))
s_norm <- rescale_to_unit_square(s, from = c(xmin, xmax, ymin, ymax))

# Validate that observations lie on the decoder's grid (within tolerance).
# Returns integer indices into dr$grid_coords (1..L).
# Errors loudly if any obs doesn't match.
obs_idx <- which_grid_points(s_norm, dr$grid_coords, tol = 1e-6)

# Snap observations to nearest grid points (lossy; opt-in).
# Returns the snapped coordinates + the grid indices.
snap <- snap_to_grid(s_norm, dr$grid_coords)
```

#### 2.3.3 Prior specification

Priors over the decoder's conditional hyperparameters are specified
explicitly per fit:

```r
prior_ls <- prior_uniform(0.05, 0.50)        # must fit inside dr$ls_trained_range
prior_ls <- prior_truncated_normal(0.2, 0.05, lower = 0.01, upper = 1.0)
```

The package validates at fit time:
- The prior's support is a subset of `dr$ls_trained_range`. Hard error
  otherwise: "decoder was trained with ℓ ∈ [0.01, 1.0]; your prior
  support extends to 1.5 — surrogate validity not guaranteed."

#### 2.3.4 The `deepRV()` formula term

```r
formula <- y ~ x1 + x2 + deepRV(s,
                                decoder    = dr,
                                obs_idx    = obs_idx,
                                ls_prior   = prior_ls,
                                by         = NULL,    # see §2.4
                                gr         = FALSE,   # see §2.4.4
                                ls_pooling = "none"   # see §2.4.5
                                )
```

The bare-minimum invocation is `deepRV(s, decoder = dr, obs_idx = obs_idx,
ls_prior = prior_ls)`. Every other argument has a sensible default but
must be silently *unambiguous*.

#### 2.3.5 The top-level fit function

```r
fit <- deeprv_brm(
  formula,
  data   = df,
  family = poisson(),
  prior  = c(prior(normal(0, 1), class = "Intercept"),
             prior(normal(0, 1), class = "b")),
  ...    # passed through to brms::brm()
)
```

This is a thin wrapper that:
1. Parses `deepRV(...)` terms out of the formula.
2. Builds `stanvars` (weights as data, `decode` function in `functions`).
3. Rewrites the formula to a brms-acceptable form (`bf(y ~ ... + mu_dr, nl = TRUE)`
   with `mu_dr` registered as a transformed parameter).
4. Calls `brms::brm()` with everything else passed through.

### 2.4 The `by` / `gr` / `ls_pooling` overloads

These are the only features that change the shape of the Stan model.
Everything else inherits from brms.

#### 2.4.1 No `by` — simple spatial smooth

```r
y ~ deepRV(s, decoder = dr, obs_idx = obs_idx, ls_prior = prior_ls)
```

Single latent `z`, single `ls`. Contributes `mu_dr[obs_idx[i]]` to `eta_i`.

#### 2.4.2 `by = factor` — group-specific smooths

```r
y ~ deepRV(s, by = region, decoder = dr, obs_idx = obs_idx,
           ls_prior = prior_ls, ls_pooling = "partial")
```

G groups (one per level of `region`). Stan model:

```stan
parameters {
  matrix[G, L] z;
  vector<lower=ls_lo, upper=ls_hi>[G] ls;       // or length 1 if pooling = "complete"
  real<lower=ls_lo, upper=ls_hi> mu_ls;          // only if pooling = "partial"
  real<lower=0> tau_ls;                          // only if pooling = "partial"
}
```

Contribution to η: `eta_i += mu_g[group_idx[i], obs_idx[i]]`.

#### 2.4.3 `by = numeric` — spatially varying coefficient (SVC)

```r
y ~ x + deepRV(s, by = x, decoder = dr, obs_idx = obs_idx, ls_prior = prior_ls)
```

Single latent `z`. Contribution to η: `eta_i += x_i * mu[obs_idx[i]]`.

The package auto-detects the type of `by` and switches between the two
semantics. **This is the same overload brms's `gp()` uses.**

#### 2.4.4 `gr = TRUE` — unique-location collapsing

When N observations sit at L_unique ≪ N grid points, compute `decode`
once per unique location. Same trick as `brms::gp(..., gr = TRUE)`.
Real win when L_unique ≪ N.

Internally: index observations by their grid point, evaluate `decode`
on the L unique locations only, then expand back to N when contributing
to `eta`.

#### 2.4.5 `ls_pooling`

Three options for how multiple groups' `ls` parameters are related:

| Value | Behavior | Stan |
|---|---|---|
| `"complete"` (default) | All groups share one `ls` | `real ls` |
| `"none"` | Groups have independent `ls`'s, each given the user's prior | `vector[G] ls` with the same prior |
| `"partial"` | `ls_g ~ N(μ_ls, τ_ls²)` truncated to trained range | `vector[G] ls; mu_ls ~ prior; tau_ls ~ <halfnormal>` |

### 2.5 Kronecker for unit_square

`unit_square` decoders are constructed at fit time from two unit_interval
decoders:

```r
dr <- load_deeprv_kron(
  domain    = "unit_square",
  grid_side = 50,            # uses unit_interval grid_size=50 decoder along each axis
  kernel    = "matern_3_2"   # axis-wise separable
)
```

Internally `dr` holds two `deepRV_decoder` references plus metadata:

```r
list(
  schema_version = 1L,
  class          = c("deepRV_decoder_kron", "deepRV_decoder"),
  domain         = "unit_square",
  grid_side      = 50L,
  L              = 2500L,                        # = grid_side^2
  grid_coords    = <2500 x 2 matrix>,            # the full 2D grid
  kernel         = "matern_3_2",
  x_decoder      = <unit_interval decoder>,      # nested deepRV_decoder
  y_decoder      = <unit_interval decoder>,      # nested deepRV_decoder
  conditionals   = c("ls_x", "ls_y"),
  ls_trained_range = c(0.01, 1.0)
)
```

**Math.** For separable kernels `K((x₁, y₁), (x₂, y₂)) = K_x(x₁,x₂) · K_y(y₁,y₂)`,
the Cholesky factor `L_full = L_x ⊗ L_y` and sampling reduces to:

```
F = L_x · Z · L_y^T
```

where `Z` is `(N × N)` iid Normal, `F` is the spatial field reshaped to
the grid. In R/Stan: apply the x-decoder along columns of `Z`, then
the y-decoder along rows.

```stan
// Z: matrix[N, N] of iid Normal (latent parameter, length L = N*N)
// step 1: apply L_x to each column of Z
matrix[N, N] mid = decode_x(Z', ls_x)';           // shape (N, N)
// step 2: apply L_y to each row of mid
matrix[N, N] F   = decode_y(mid, ls_y);           // shape (N, N)
// flatten back to length L for the linear predictor.
```

**Supported kernels for unit_square (v0.1):** `matern_1_2`, `matern_3_2`,
`matern_5_2`, `rbf` — interpreted as axis-wise separable. **Isotropic
versions are out of scope; ship them in v0.2 with monolithic 2D
decoders.**

User error to surface:
- Loading `unit_square` for a kernel name that we only have a separable
  variant of: print a clear warning that the model is axis-wise
  Matérn (not isotropic Matérn).

### 2.6 Space-time term

```r
fit <- deeprv_brm(
  y ~ x1 + deepRV_st(
        decoder_space = load_deeprv_kron("unit_square", 50, "matern_3_2"),
        decoder_time  = "rw",                                # see below
        ls_x_prior    = prior_uniform(0.05, 0.5),
        ls_y_prior    = prior_uniform(0.05, 0.5),
        sigma_t_prior = prior_exp(1),                        # for rw
        obs_idx       = obs_idx,                             # spatial grid index per obs
        time_idx      = t_idx                                # time index per obs
       ),
  data   = df,
  family = poisson()
)
```

`decoder_time` accepts one of three forms:

| `decoder_time =` | Prior on time axis | Stan implementation |
|---|---|---|
| `"rw"` | RW(σ_t) | Cumulative sum: `F_t = F_{t-1} + σ_t · z_t` |
| `"ar1"` | AR(1) with autocorrelation φ | Closed-form Cholesky |
| `<a unit_interval decoder>` | GP with Matérn / RBF in time | Standard `decode()` call along time axis |

The Stan model has a 3D latent `Z` of shape `(T, L_x, L_y)`. Apply the
three 1D operations sequentially along each axis. Total per-leapfrog
work: `T·N` forwards of `decode_x` (small) + `T·N` of `decode_y` (small)
+ `N²` RW updates or 1 `decode_t` call.

### 2.7 Post-processing

Each requires an S3 method registered for the `deepRV` term.

#### 2.7.1 `posterior_predict.deepRV` / `posterior_epred.deepRV`

- **At training grid points:** standard. Use the posterior draws of `z`,
  `ls`, and the decoder weights (data) to compute `mu` and then the
  family's `posterior_predict`.
- **At new locations:** **error** with a clear message:
  `"deepRV decoders only support prediction at training grid points.
   Use brms::gp() for new-location prediction, or snap_to_grid() to
   round new observations to grid points."`

#### 2.7.2 `conditional_effects.deepRV`

Plot the posterior spatial field. For 1D: line plot of the posterior
mean and credible interval over `dr$grid_coords`. For 2D: heatmap.
For `by = group`: faceted by group.

This is the single highest-priority post-processing feature. Without
it, the package's UX is hollow.

#### 2.7.3 `ranef.deepRV` / `coef.deepRV`

For `by = factor` runs, extract per-group `mu_g` and `ls_g` posterior
summaries. Mirror brms's existing `ranef()` output structure as closely
as possible.

#### 2.7.4 `loo`, `waic`, `pp_check`, `fitted`

These work for free since they go through `posterior_predict` and the
standard `log_lik` machinery. No new methods needed.

### 2.8 Training pipeline

Owner: a Python script in the dl4bi repo, NOT shipped with the R package.

```bash
uv run python scripts/train_decoders.py --config configs/v0.1.yaml
```

`v0.1.yaml`:

```yaml
arch: MLPDeepRV
arch_version: "1.0.0"
domain: unit_interval
ls_trained_range: [0.01, 1.0]
training:
  steps: 100000
  batch: 32
  lr: 0.001
  optimizer: adamw
  weight_decay: 0.01
  clip_global_norm: 3.0
  seed: 0
grids: [5, 10, 20, 50, 100, 200, 500, 1000]
kernels: [matern_1_2, matern_3_2, matern_5_2, rbf]
output_dir: artifacts/decoders/v0.1/
```

Outputs 32 `.rds` files plus a `manifest.json` containing the
fingerprint → filename map. A separate script bundles these into the
R package (`inst/extdata/decoders/`) or uploads to a GitHub release for
download.

**Reproducibility:** seed-controlled, deterministic on the same JAX
version. The arch_version bumps if the Stan port changes; old `.rds`
files become unusable and the loader errors with a clear message.

### 2.9 R package skeleton

```
brms.deeprv/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── load_decoder.R         # load_deeprv, load_deeprv_kron
│   ├── coords.R               # rescale_*, which_grid_points, snap_to_grid
│   ├── priors.R               # prior_uniform, prior_truncated_normal, validation
│   ├── formula.R              # deepRV(), deepRV_st() formula terms
│   ├── stanvars.R             # build_stanvars(), Stan code injection
│   ├── fit.R                  # deeprv_brm() wrapper
│   ├── post_processing.R      # posterior_predict.deepRV, conditional_effects.deepRV, ranef.deepRV
│   ├── utils.R                # fingerprinting, validation
│   └── zzz.R                  # onLoad hook to verify catalog
├── inst/
│   ├── extdata/
│   │   ├── decoders/          # the 32 .rds files (or downloaded on demand)
│   │   └── manifest.json      # fingerprint → filename
│   └── stan/
│       ├── decode_mlp.stan    # the `decode` function (Stan, not C++)
│       └── model_template.stan # the parameters/model blocks
├── tests/testthat/
│   ├── test-load.R
│   ├── test-coords.R
│   ├── test-formula.R
│   ├── test-recovery.R        # end-to-end Poisson recovery on the Matern fixture
│   └── test-by.R              # group/SVC mode
└── vignettes/
    ├── 01-getting-started.Rmd
    ├── 02-spatial-varying-coefficients.Rmd
    └── 03-space-time.Rmd
```

### 2.10 Testing

| Test | Method | Pass criterion |
|---|---|---|
| Forward parity | Compare `decode()` output between rstan (via `expose_stan_functions`) and the JAX reference saved alongside each decoder | Max abs diff < 1e-5 |
| HMC sanity | 100 iter NUTS on a tiny fixture | Rhat < 1.05, 0 divergences |
| Recovery (end-to-end) | Reproduce the `run_hmc.R` test case from the prototype | 90% CI covers truth, n_eff > 100 |
| `by = factor` recovery | 3-group synthetic fixture | Per-group posterior means within ±1 of truth |
| `by = numeric` (SVC) recovery | Synthetic SVC fixture | Same |
| `gr = TRUE` correctness | Same posterior as the explicit-obs version | Posterior samples agree to MCSE |
| Kronecker forward match | 2D field vs. the JAX direct computation `L_x · Z · L_y^T` | Max abs diff < 1e-5 |

---

## 3 · v0.2 — Sketch

### 3.1 Monolithic 2D decoders for isotropic kernels

Ship 4 monolithic `unit_square` decoders for grid_side ∈ {5, 10, 20, 50}
(L up to 2500) with truly isotropic kernels (Matérn over Euclidean
distance). These can't be Kronecker-composed.

Compute cost is bounded by L² weight matrices, so we cap at 50×50.
Cost per HMC iteration scales as the MLP at this L. Sampling time at
L=2500 will be in the few-minutes range. Acceptable as v0.2 add-on.

### 3.2 Distributional models

```r
brm(bf(y ~ deepRV(s), sigma ~ deepRV(s)), data = df, family = gaussian())
```

Spatially varying scale. Same machinery as `brms::gp()` on distributional
parameters. Implementation: the formula parser needs to recognize which
sub-formula a `deepRV` lives in and contribute to the right `eta`.

### 3.3 More kernels

Periodic, spectral-mixture, separable spatiotemporal kernels with
non-trivial cross-time structure. Each kernel triples the catalog size.
Add on demand.

### 3.4 More domains

- `unit_cube` (3D, [0, 1]³) for volumetric data (rare in applied work)
- `unit_disk` / `sphere` for naturally radial data

These require new decoder training pipelines and are speculative.

### 3.5 Custom retraining hook

For users with a Python environment, an optional `train_deeprv()`
function via reticulate that produces a custom decoder. Stays optional.
Don't make this a default path — it complicates the "CPU-only" promise.

### 3.6 Nested random spatial fields

`deepRV(s, by = region/site)` — multi-resolution spatial modeling
(coarse field at region level + fine field at site level). Real
modeling power but identifiability concerns. v0.2 nice-to-have.

### 3.7 gMLP variant

For users who genuinely need the better surrogate quality of `gMLPDeepRV`
and accept ~80 min/fit, expose a `arch = "gmlp"` option in `load_deeprv()`.
Use the external C++ path already prototyped. Document loudly that this
is "research-grade, not interactive."

---

## 4 · Lessons learned from the prototype (read this before writing code)

These are non-obvious gotchas surfaced during the prototype work. Carry
them forward.

### 4.1 GELU defaults differ

**`jax.nn.gelu` defaults to `approximate=True`** (tanh form), not the
exact `x · Φ(x)` form. `flax.linen.gelu` re-exports `jax.nn.gelu` so the
trained decoders use the tanh approximation:

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

The Stan port (`gmlp_decode.stan`) and numpy mirror must match. **This
was a real bug in the prototype** — initial Stan implementation used
`x · Phi(x)` and forward-pass mismatch was ~1e-3 instead of ~1e-6.

### 4.2 Flax param tying in `SpatialGatingUnit`

`SpatialGatingUnit` has `norm: nn.Module = nn.LayerNorm()` as a
class-level default. Flax detects the shared instance across multiple
`gMLPBlock` invocations and **ties the LayerNorm parameters**. Only the
first block's `SpatialGatingUnit_0/norm` appears in the param tree;
subsequent blocks use the same params.

Any Stan / C++ port of `gMLPDeepRV` must apply the same `(scale, bias)`
in both blocks' SGU. **Only relevant if v0.2 adds the gMLP variant.**

### 4.3 R `unlist()` collapses length-1 lists to scalars

When loading JSON weights into Stan data, a `vector[1]` declaration
errors with "dims declared=(1); dims found=()" because `unlist(list(0.5))`
returns a bare numeric.

**Fix:** wrap with `as.array(unlist(v))`. Already in the prototype's
`run_gmlp_hmc.R`. Carry forward.

### 4.4 External C++ via `--allow-undefined` is a trap for small models

We measured `MLPDeepRV` external C++ as **slower** than all-Stan (57 s
vs. 42 s sampling at L=256). The materialization tax (Eigen::Map →
Eigen::Matrix copies, intermediate matrices from stan::math ops) exceeds
the AD-tape savings for a forward pass with only a few ops.

**Decision:** v0.1 uses **all-Stan only** for `MLPDeepRV`. The external
C++ path stays around for the v0.2 gMLP variant only.

### 4.5 Stan template arg deduction doesn't apply Eigen implicit conversions

Stan passes parameters as `Eigen::Matrix<T, -1, -1>` and data as
`Eigen::Map<Eigen::Matrix<...>>`. A C++ template that takes `const
Eigen::Matrix<T, -1, -1>&` won't deduce from a `Map`. Fix: take any
`const T&` per arg and materialize inside the function. **Only relevant
if v0.2 revives the external C++ path.**

### 4.6 Apt's `r-cran-bh` is a metapackage, not a CRAN-style BH

On Debian/Ubuntu, `apt install r-cran-bh` doesn't put headers under
`system.file("include", package = "BH")`. rstan errors with "Boost not
found." Fix: symlink `/usr/include/boost` into the BH package include
directory. This belongs in a `setup_environment.sh` somewhere.

### 4.7 Stan compile cache (`auto_write = TRUE`)

`rstan_options(auto_write = TRUE)` writes a `.rds` next to each `.stan`
file after first compile. Subsequent fits skip the 1–2 min compile.
The cache becomes stale on Stan code changes — must be invalidated
when bumping `arch_version`.

### 4.8 Python ↔ R float formatting disagrees everywhere except `%.8f`

When a fingerprint must agree across Python and R, you cannot use:

- `repr(float(x))` / `str(float(x))` — Python writes `1.0`, R has no
  equivalent (`as.character(1.0)` gives `"1"`).
- `formatC(x, digits=17, format="g")` — pads with leading spaces and
  produces different significant-figure counts than Python's `repr`.
- `jsonlite::toJSON(1.0)` — writes `1`; `json.dumps(1.0)` writes `1.0`.

`f"{x:.8f}"` in Python and `sprintf("%.8f", x)` in R produce identical
bytes for all finite doubles within ~8 fractional digits of precision.
This is what the v0.1 fingerprint uses (§2.2). If you ever need more
precision in the fingerprint, you'll have to special-case it.

### 4.9 The forward-pass match check is the most important test

When porting an architecture, the Stan implementation can be silently
wrong in ways that still produce plausible-looking HMC output (Rhat=1,
no divergences, posterior covers truth). The only reliable way to
catch this is the JAX-vs-Stan forward parity check at the per-element
level. Every new decoder ships with golden test cases (a handful of
`(z, cond, jax_output)` triples) and the package's test suite verifies
the match at install time.

---

## 5 · Critical limitations to document loudly

These are constraints the package cannot work around and must be
prominent in user-facing docs:

1. **No new-location prediction.** `MLPDeepRV` is trained for a specific
   grid. `posterior_predict` at unseen locations is impossible. Users
   who need this go to `brms::gp()`.
2. **No irregular geometries.** The package handles only data on the
   shipped grids. Irregular point patterns require either snapping
   (lossy) or `brms::gp()`.
3. **Priors must fit within trained range.** Inference with `ls`
   outside the trained range silently uses the decoder in a regime
   where its accuracy isn't guaranteed. The package errors when the
   user's prior support exceeds the trained range.
4. **Separable kernels only for unit_square in v0.1.** True isotropic
   2D kernels arrive in v0.2.

---

## 6 · Performance budget

Per-fit wall-time targets at L=256, 2 chains × 1000 NUTS iter,
`adapt_delta = 0.95`:

| Feature | Target wall time |
|---|---|
| Bare `deepRV(s)` | < 60 s |
| `deepRV(s, by = group)` for G ≤ 100 | < 120 s |
| `deepRV(s, by = numeric)` (SVC, 1 coef) | < 60 s |
| `deepRV(s, gr = TRUE)` (unique locs ≤ 100) | < 30 s |
| `deepRV_st(...)` for T = 50 | < 5 min |
| Kronecker `unit_square` at side = 50 (L = 2500) | < 10 min |

If any of these exceed the target by > 2×, surface a clear warning at
fit time pointing the user to lower `adapt_delta` or use fewer iterations.

---

## 7 · Out of scope (v0.1 AND v0.2)

- `me()` measurement error in spatial coordinates
- `mi()` missing-data imputation
- `car()` / `sar()` areal models
- `mo()` monotonic effects
- Splines (`s()`, `t2()`)
- Non-Euclidean domains (sphere, hyperbolic)
- Custom user-trained decoders (use `dl4bi` directly)
- Anisotropic non-separable kernels (no `cov="auto"`)

---

## 8 · Implementation order

Recommended order for the v0.1 build:

1. ~~**Training pipeline**~~ — **done in `77286a6`.** YAML config →
   `train_decoders.py` → per-decoder JSON → `pack_decoders.R` → canonical
   `.rds` + `manifest.json`. Smoke run (L=10, 1000 steps, 2 decoders)
   round-trips end-to-end with forward-match at 1.4e-7 against the JAX
   reference. The full 32-decoder catalog still has to be **trained** —
   that's the wall-clock cost (~1 GPU-day / ~1 CPU-week), but the
   pipeline is built.
2. ~~**R package skeleton + `load_deeprv()`**~~ — **done.** Package at
   `brms.deeprv/` with `DESCRIPTION`, `NAMESPACE`, `R/load_decoder.R`
   (`load_deeprv`, `load_deeprv_kron`, S3 `print` methods), `R/utils.R`
   (`deeprv_fingerprint`, `validate_decoder`), `inst/extdata/decoders/`
   seeded with the smoke artifacts, `inst/stan/decode_mlp.stan` (the
   forward function), `man/*.Rd`. `R CMD check --no-manual` passes with
   zero notes/warnings/errors; 42 testthat assertions green including
   the forward-parity check from step 4 below.
3. **Coordinate helpers** — `rescale_to_unit_interval()`, `which_grid_points()`,
   `snap_to_grid()`. ~1 day.
4. ~~**Forward-match tests**~~ — **done as part of step 2** —
   `tests/testthat/test-forward.R` loops over every shipped decoder,
   exposes `decode_mlp.stan::decode` via `rstan::expose_stan_functions`,
   and compares JAX-saved golden cases to < 1e-5 (currently passing
   at ~1e-7 for both smoke decoders). Step 4's role in the build order
   is now "extend the parity test to cover every decoder in
   `configs/v0.1.yaml` once the catalog is trained."
5. **`deepRV()` formula term + `deeprv_brm()` wrapper** — basic spatial only
   (no `by`, no `gr`). Reuses `run_hmc.R` patterns. **Done for v0.1.**
   `R/formula.R`: `deepRV(...)` is a marker that captures its arguments;
   `parse_deeprv_formula()` walks the RHS, extracting a single deepRV
   term and returning the residual. `R/priors.R`: `prior_uniform(lo, hi)`
   + `validate_prior_in_range()` (the section 2.3.3 "prior must fit the
   trained range" check). `R/stancode.R`: builds a complete Stan
   program inlining `inst/stan/decode_mlp.stan`'s functions block plus
   data/parameters/transformed parameters/model blocks for the
   Poisson or Gaussian family with an arbitrary fixed-effect design
   matrix `X`. `R/fit.R`: `deeprv_brm()` parses, validates, builds
   `model.matrix(~ rhs, data)`, assembles standata, compiles, and
   samples via `rstan::stan()`. Remaining gates: `by`, `gr`,
   `ls_pooling != "complete"`, families other than `poisson` /
   `gaussian`. Each is a `stop()` in `deeprv_brm()` (search for the
   reject messages) naming what to lift.
6. ~~**`posterior_predict.deepRV`, `conditional_effects.deepRV`**~~ —
   **done for v0.1.** `R/post_processing.R`: `posterior_eta_draws()`
   is the underlying primitive (R-side batched forward, matches
   Stan's `mu` to < 1e-4). `posterior_predict.deeprv_fit` and
   `posterior_epred.deeprv_fit` register against the rstantools
   generics. `R/conditional_effects.R`: `conditional_effects()` is a
   local generic (no brms dep) returning a data frame per deepRV term
   with `(grid_index, s, estimate, lower, upper)`; ships with a
   base-graphics `plot()` method (line + CI ribbon). 2D Kronecker
   heatmap deferred to v0.2 along with the Kronecker fit path.
7. **`by = factor`, `by = numeric`, `ls_pooling`** —
   **`by = numeric` (SVC) and `by = factor` (groups) done with
   `ls_pooling = "complete"`.** SVC: Stan likelihood uses
   `x_by .* mu[obs_idx]`. Factor: `matrix[G, L] z; matrix[G, L] mu`
   via per-group decode loop, single shared `ls`, spatial gathered
   as `mu[group_idx[n], obs_idx[n]]`. `R/fit.R::classify_by()` routes
   `NULL` / `numeric` / `factor|character`. Post-processing
   (`posterior_eta_draws`, `conditional_effects`, plot) all handle
   factor mode (conditional_effects emits one row per (group, grid)
   and the plot method facets by group). Remaining: `ls_pooling` !=
   `"complete"` (hierarchical or independent per-group length scales)
   and the combination of `by = factor` with `by = numeric` (which
   the design doc doesn't ask for).
8. **`gr = TRUE`** — ~2 days.
9. **`load_deeprv_kron()`, `deepRV()` accepting `deepRV_decoder_kron`** —
   Stan code path for the Kronecker composition. ~1 week.
10. **`deepRV_st(...)` + RW/AR1/decoder time priors** — ~1 week.
11. **Vignettes, polishing, CRAN submission prep.** ~1 week.

**Total v0.1 estimate: ~6 person-weeks of engineering + GPU/CPU time for
the catalog.**

---

## 9 · Handoff bundle

Branch: `claude/deeprv-rstan-integration-VDtiw`.
Steps 1, 2, 4, and the MVP of step 5 are done. Next is widening
step 5 (covariates, `by`, `gr`, families other than Poisson) and/or
step 3 (coordinate helpers) and/or step 6 (post-processing).

### What's in place

| Path | Role |
|---|---|
| `benchmarks/vae/rstan/train_decoders.py` | Step 1 trainer |
| `benchmarks/vae/rstan/pack_decoders.R` | Step 1 JSON → .rds packer |
| `benchmarks/vae/rstan/verify_rds.R` | Step 1 install-time forward check |
| `brms.deeprv/DESCRIPTION`, `NAMESPACE`, `LICENSE`, `.gitignore` | Step 2 package metadata |
| `brms.deeprv/R/utils.R` | `deeprv_fingerprint()`, `validate_decoder()`. **The Python and R fingerprint formulas live here and in `train_decoders.py` + `pack_decoders.R` — keep them in lock-step.** |
| `brms.deeprv/R/load_decoder.R` | `load_deeprv()`, `load_deeprv_kron()`, S3 `print` methods. Decoder dir overridable via `options(brms.deeprv.decoder_dir)` or `BRMS_DEEPRV_DECODER_DIR` env var. |
| `brms.deeprv/R/zzz.R` | `.onLoad`: env-var → option bridge |
| `brms.deeprv/R/priors.R` | `prior_uniform()`, `validate_prior_in_range()`. v0.2 extension point for `prior_truncated_normal()` and friends. |
| `brms.deeprv/R/formula.R` | `deepRV(...)` marker + `parse_deeprv_formula()` walker. |
| `brms.deeprv/R/stancode.R` | `build_stancode()`, `build_standata()`. Inlines `inst/stan/decode_mlp.stan`'s functions block. **MVP only emits the Poisson/intercept/uniform-ls path** — every other branch is a `stop()` in `deeprv_brm()`. |
| `brms.deeprv/R/fit.R` | `deeprv_brm()` glue and the `deeprv_fit` S3 class. |
| `brms.deeprv/inst/extdata/decoders/` | Smoke catalog seed (2 unit_interval_10 decoders + manifest.json). Replace with the v0.1 catalog before release. |
| `brms.deeprv/inst/stan/decode_mlp.stan` | The shipped `decode` function (Stan, no model block). Bumping this REQUIRES bumping `arch_version` in the catalog. |
| `brms.deeprv/man/*.Rd` | Hand-written; switch to roxygen2 generation once that dep is installed. |
| `brms.deeprv/tests/testthat/test-load.R` | 27 assertions: manifest hit/miss, kernel/domain validation, fingerprint tamper detection, Python ↔ R fingerprint byte-equality. |
| `brms.deeprv/tests/testthat/test-forward.R` | rstan-vs-JAX forward parity across every shipped decoder. 15 assertions, ~24 s including Stan compile. |
| `brms.deeprv/tests/testthat/test-priors.R` | 10 assertions: `prior_uniform()` bounds checking + trained-range validation. |
| `brms.deeprv/tests/testthat/test-formula.R` | 14 assertions: parser pulls a clean spec from `y ~ deepRV(...)`, retains other RHS terms as residual, errors on zero/multiple deepRV terms. |
| `brms.deeprv/tests/testthat/test-fit.R` | 12 assertions: end-to-end Poisson fit on the smoke fixture, ~50 s sampling, no divergences. Skipped under `NOT_CRAN` unset. |

### How to verify

```bash
# Cheap sanity (~1 min, hits the rstan Boost env): the legacy prototype
# check is independent of the package and is the first thing to run.
Rscript benchmarks/vae/rstan/check_match.R

# Package check (~10 s after Stan cache warm):
R CMD build brms.deeprv && R CMD check --no-manual brms.deeprv_*.tar.gz
```

If `check_match.R` fails, the dev env is wrong — see §4.6 (Boost headers
on Ubuntu) before anything else.

### Decisions made along the way

- **Package location:** top-level `brms.deeprv/` in this repo. Split
  into a separate repo only when CRAN-bound.
- **Catalog timing:** smoke fixtures first, full v0.1 catalog later.
  Path 2 in the previous handoff. The smoke `.rds` files in
  `inst/extdata/decoders/` are dev fixtures, not shippable artifacts;
  bumping `arch_version` is fine because no users exist yet.
- **License:** MIT, declared in `DESCRIPTION` and `LICENSE`.
- **Docs:** hand-written `man/*.Rd` for now. If you have `roxygen2`
  installed and want to regenerate, the source comments are still in
  the R files.

### What the next agent should pick up

1. **Decide whether to train the v0.1 catalog now** (`configs/v0.1.yaml`,
   ~1 CPU-week unattended) or keep using smoke fixtures. The fingerprint
   pipeline is locked, so the package will accept the real catalog the
   moment it's copied into `inst/extdata/decoders/`. Until then, the
   end-to-end test in `test-fit.R` only confirms wiring, not recovery —
   the run_hmc.R-style "90% CI covers truth" test waits for real
   decoders.
2. **Widen step 5 by lifting MVP gates one at a time.** Each gate is a
   `stop()` in `deeprv_brm()` (search for "MVP:"); lifting it means
   editing both that check and `build_stancode()`/`build_standata()`.
   Suggested order, smallest first:
   - **Covariates on the RHS.** Add a `model.matrix(parsed$rhs, data)`
     call, push `X` and `K` into standata, add `vector[K] b; beta0 +
     X * b` to the Stan program. Brms-style `prior(normal(0,1),
     class = "b")` priors get translated to Stan prior statements.
   - **Other families.** `gaussian()` and `bernoulli()` are obvious
     next targets; family branching lives entirely in
     `build_stancode()`.
   - **`by = factor` and `by = numeric`.** The Stan side changes
     shape (matrix `z`, vector `ls`); see section 2.4.2 / 2.4.3.
   - **`gr = TRUE`.** Pure data-side change; see section 2.4.4.
3. **§8 step 3 — coordinate helpers** in `R/coords.R`:
   `rescale_to_unit_interval()`, `rescale_to_unit_square()`,
   `which_grid_points()`, `snap_to_grid()`. Small, testable, no Stan
   involvement. Can land in parallel with step 5 widening.
4. **§8 step 6 — `posterior_predict.deepRV`, `conditional_effects.deepRV`.**
   The `deeprv_fit` object already carries the stanfit and decoder;
   adding these S3 methods is mostly a matter of extracting `z`/`ls`
   draws, running them through `decode_mlp.stan` (R-side or via
   `expose_stan_functions()`), and shaping the result like brms's
   existing output.

### Open questions to batch back to the user

- **CRAN strategy.** The package will ship 32 binary `.rds` files
  (~tens of MB total). CRAN's 5 MB tarball limit forces either a
  download-on-demand model (download from a GitHub release on first
  `load_deeprv()`) or non-CRAN distribution.
- **Whose JAX version is the catalog frozen against?** Bumping JAX
  may shift weights below the 1e-5 forward-match tol; the
  `arch_version` field exists to gate this.
- **Where does the brms boundary fall?** `deeprv_brm()` is described
  as a thin wrapper, but the formula-rewrite path in §2.3.5 looks
  load-bearing — confirm with the user whether to lean on
  `brms::stanvars` for the data + function injection, or fork the
  brms code-generation path.
