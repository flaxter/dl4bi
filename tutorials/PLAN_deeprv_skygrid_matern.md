# Tutorial Plan: Matérn GP Prior for Ne(t) with DeepRV

## Motivation

The standard Skygrid coalescent (Gill et al. 2013) uses a **GMRF** (first-order random walk) prior on log Ne(t). The GMRF is intrinsic (non-stationary) — it penalises adjacent differences but has no explicit length-scale or marginal variance.

**Nobody has published a stationary GP prior on Ne(t) over the time grid** — not Suchard's group, not anyone else. Palacios & Minin (2013) used an integrated Brownian motion GP (non-stationary). Monti et al. (2026, arXiv:2602.06148) use a GP over *covariates* but keep GMRF for the time axis — implemented in BEAST X.

This tutorial replaces GMRF with a **Matérn 3/2 GP**, using **DeepRV** to avoid O(M³) Cholesky at each MCMC step.

## Setup

- **Branch**: `claude/deeprv-nonnegative-tutorial-QQCvh`
- **Skygrid code**: already cherry-picked onto that branch (`dl4bi/coalescent/skygrid.py`, `hcv_egypt.py`)
- **Dataset**: HCV Egypt — 63 sequences, 62 coalescent heights, all sampled in 1993
- **File to create**: `tutorials/deeprv_skygrid_matern.ipynb`

## Key Parameters

```python
M       = 20                                    # grid intervals
CUTOFF  = float(HEIGHTS[-1]) * 1.1              # ≈ 305.76 years
grid_full   = jnp.linspace(0, CUTOFF, M + 1)   # 21 points
grid        = grid_full[1:-1]                   # (19,) interior change-points
grid_times  = ((grid_full[:-1] + grid_full[1:]) / 2)[:, None]  # (M, 1) midpoints → DeepRV "s"
times_flat  = grid_times[:, 0]                  # (M,) for kernel computation
```

## Matérn 3/2 Kernel

```python
@jit
def matern32_kernel(times, log_sigma2, log_ell, jitter=1e-5):
    sigma2 = jnp.exp(log_sigma2)
    ell    = jnp.exp(log_ell)
    d = jnp.abs(times[:, None] - times[None, :])
    r = jnp.sqrt(3.0) * d / ell
    K = sigma2 * (1.0 + r) * jnp.exp(-r)
    return K + jitter * jnp.eye(times.shape[0])

@jit
def gp_cholesky(times, log_sigma2, log_ell):
    return jnp.linalg.cholesky(matern32_kernel(times, log_sigma2, log_ell))
```

## Training Priors (for DeepRV training data generation)

```python
LOG_SIGMA2_MEAN = 0.0;   LOG_SIGMA2_STD = 2.0    # σ² ∈ ~[0.02, 55]
LOG_ELL_MEAN    = 4.336;  LOG_ELL_STD   = 1.0     # ℓ centered at cutoff/4 ≈ 76 yr
```

## Training Data Generator

Each batch:
1. Sample `log_sigma2`, `log_ell` from the training priors
2. Compute `L = cholesky(K(σ², ℓ))` — shape (M, M)
3. Sample `z ~ N(0, I_M)` — shape (batch, M)
4. Compute `f = einsum("ij,bj->bi", L, z)` — shape (batch, M)

Yield `{"s": grid_times, "z": z, "conditionals": [log_sigma2, log_ell], "f": f}`

## DeepRV Training

```python
nn_model  = gMLPDeepRV(num_blks=3)
N_STEPS   = 8_000
optimizer = optax.chain(
    optax.clip_by_global_norm(3.0),
    optax.adamw(cosine_annealing_lr(N_STEPS, 1e-3), weight_decay=1e-2),
)
```

Use `Callback` at interval=200 to record RMSE on a fixed held-out batch (σ²=2, ℓ=cutoff/4).
Store snapshots at steps {500, 2000, 5000, 8000}.

After training: `surrogate_decoder = generate_surrogate_decoder(state, nn_model)`

## Surrogate Decoder Call Convention

```python
# Training/visualization (returns VAEOutput with .f_hat):
f_hat = nn_model.apply(
    {"params": state.params, **state.kwargs},
    z_batch, jnp.array([log_sigma2, log_ell]), s=grid_times,
    rngs={"extra": rng},
).f_hat.squeeze(-1)   # (batch, M)

# Inference (returns raw tensor):
log_thetas = surrogate_decoder(z[None], conditionals, s=grid_times)[0, :, 0]  # (M,)
```

## Visualizations

### 1. Training curve + trajectory snapshots
- **Top row**: RMSE vs training step (log scale y), vertical lines at snapshot steps
- **Bottom row**: One subplot per snapshot showing 10 GP trajectories:
  - Blue solid = exact L·z
  - Red dashed = DeepRV f̂
  - x-axis = years before present (grid midpoints)

### 2. Scatter: f̂ vs exact across (σ², ℓ) pairs
Test 4 hyperparameter pairs not seen verbatim during training:
- (σ²=0.5, ℓ=cutoff/8), (1.0, cutoff/4), (2.0, cutoff/2), (4.0, cutoff)
- For each: scatter f̂[idx] vs f_exact[idx] at grid index M//2, 300 z samples

## NumPyro Models

### GMRF Baseline

```python
def gmrf_model(node_heights, sampling_times, grid):
    precision = numpyro.sample("precision", dist.Gamma(0.001, 0.001))
    log_thetas = numpyro.sample(
        "log_thetas", dist.Normal(0.0, 10.0).expand([M]).to_event(1))
    numpyro.factor("gmrf", gmrf_log_prob(log_thetas, precision))
    numpyro.factor("coalescent",
        skygrid_coalescent_log_prob(log_thetas, grid, node_heights, sampling_times))
```

### Matérn GP + DeepRV

```python
def matern_deeprv_model(surrogate_decoder, node_heights, sampling_times, grid):
    log_sigma2 = numpyro.sample("log_sigma2", dist.Normal(0.0, 2.0))
    log_ell    = numpyro.sample("log_ell",    dist.Normal(4.336, 1.0))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([M]).to_event(1))

    conditionals = jnp.array([log_sigma2, log_ell])
    log_thetas = numpyro.deterministic(
        "log_thetas",
        surrogate_decoder(z[None], conditionals, s=grid_times)[0, :, 0],
    )
    numpyro.factor("coalescent",
        skygrid_coalescent_log_prob(log_thetas, grid, node_heights, sampling_times))
```

## NUTS Configuration

```python
def run_nuts(rng_key, model, label, **kwargs):
    kernel = NUTS(model, max_tree_depth=10, init_strategy=init_to_median(num_samples=15))
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    mcmc.run(rng_key, **kwargs)
    mcmc.print_summary(exclude_deterministic=True)
    return mcmc
```

## Results Plot

```python
SAMPLING_YEAR = 1993.0

def plot_ne(ax, mcmc, label, color):
    log_thetas = mcmc.get_samples()["log_thetas"]   # (S, M)
    ne = np.exp(np.array(log_thetas))
    median = np.median(ne, axis=0)
    lo, hi = np.percentile(ne, [2.5, 97.5], axis=0)
    cal = SAMPLING_YEAR - np.array(times_flat)

    ax.fill_between(cal, lo, hi, alpha=0.25, color=color)
    ax.plot(cal, median, color=color, lw=2, label=label)

fig, ax = plt.subplots(figsize=(11, 5))
plot_ne(ax, mcmc_gmrf, "GMRF", "steelblue")
plot_ne(ax, mcmc_drv,  "Matérn 3/2 GP + DeepRV", "tomato")
ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("Calendar year"); ax.set_ylabel("Ne(t)")
ax.legend()
```

Also plot posterior histograms of σ² and ℓ from the Matérn model.

## Colab Setup Cell

Same pattern as `deeprv_nonneg_bivariate.ipynb`:
```python
# Clone from branch claude/deeprv-nonnegative-tutorial-QQCvh
# pip install: flax>=0.12, hydra-core>=1.3, numpyro, jraph, einops,
#              orbax-checkpoint, scoringrules,
#              git+https://github.com/MLGlobalHealth/sps.git
```

## Imports

```python
from dl4bi.coalescent.skygrid import gmrf_log_prob, skygrid_coalescent_log_prob
from dl4bi.coalescent.hcv_egypt import HEIGHTS, SAMPLING_TIMES
from dl4bi.core.train import Callback, cosine_annealing_lr, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
```

## Summary Table (final markdown cell)

| | GMRF (baseline) | Matérn 3/2 GP + DeepRV |
|---|---|---|
| **Prior** | Intrinsic first-order RW | Stationary Matérn 3/2 |
| **Hyperparameters** | τ (precision) | σ² (variance), ℓ (length-scale) |
| **Length-scale** | None — all adjacent changes penalised equally | Explicit ℓ inferred from data |
| **Cholesky cost per MCMC step** | O(M) tridiagonal | O(1) neural forward pass |
| **Training cost** | None | O(N_STEPS) one-time |
