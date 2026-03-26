# Handoff: Skygrid coalescent in NumPyro

## What we're building

A NumPyro reimplementation of the **Skygrid** coalescent model for inferring
effective population size Nₑ(t) over time from viral sequences, benchmarked on
the classic **HCV Egypt dataset**. No DeepRV yet — get the NumPyro port working
first, then drop in a surrogate later.

## Reference implementation

**torchtree** by Mathieu Fourment (co-authors: Suchard, Matsen IV)
- Repo: https://github.com/4ment/torchtree
- Paper: Fourment et al. (2026), *Systematic Biology* 75(1):39
- Language: Python + PyTorch, fixed-tree inference
- Relevant classes: `PiecewiseConstantCoalescentGridModel` (Skygrid),
  `PiecewiseConstantCoalescentModel` (Skyride)
- HCV Egypt data: in `torchtree-experiments` repo (https://github.com/4ment/torchtree-experiments)
- Last release: v1.0.2, July 2024. Actively maintained.

The plan is to **fork torchtree**, study its coalescent likelihood implementation,
and rewrite the model in NumPyro/JAX.

## The model

### Data
- 63 HCV RNA sequences (E1 region), all sampled in Egypt in 1993 (contemporaneous)
- The input to inference is a **fixed dated phylogeny** (Newick/NEXUS), not raw sequences
- The tree gives a list of **coalescent event times** (when lineages merge)

### Likelihood: coalescent
Given a fixed tree with n tips, there are n-1 coalescent events at times
t₁ < t₂ < ... < tₙ₋₁ (going backwards in time). Between consecutive events,
k lineages are present. The waiting time between events is exponential with rate
C(k) / Nₑ(t), where C(k) = k(k-1)/2.

For the Skygrid, time is divided into a fixed grid of M intervals with
endpoints 0 = s₀ < s₁ < ... < sₘ. Within each interval Nₑ is piecewise
constant at value exp(γₖ). The log-likelihood is:

    log p(tree | γ) = Σ_intervals [ -C(k) * Δt / exp(γ) ] + Σ_coalescent_events [ -γ ]

### Prior: GMRF on log Nₑ
The Skygrid prior on γ = (γ₁, ..., γₘ) is a first-order Gaussian Markov
Random Field (GMRF), equivalent to a random walk / Matérn-1/2 on the grid:

    p(γ | τ) ∝ τ^{(M-1)/2} exp( -τ/2 · Σₖ (γₖ₊₁ - γₖ)² / δₖ )

where δₖ = sₖ₊₁ - sₖ are the grid spacings and τ ~ Gamma(0.001, 1000) is a
global precision (smoothness) hyperparameter.

This is NOT a full GP with a stationary kernel — it's a sparse precision-matrix
prior. The Cholesky of the precision matrix (not covariance) is tridiagonal and
cheap. **This is also exactly the structure DeepRV could later emulate** when M
is large.

### Inference
MCMC over (γ₁, ..., γₘ, τ). The tree is fixed. With NUTS in NumPyro this
should be straightforward.

## Key references
- **Skygrid**: Gill et al. (2013), *MBE* 30(3):713
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3563973/
- **Skyride**: Minin, Bloomquist & Suchard (2008), *MBE* 25(7):1459
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3302198/
- **HCV Egypt**: Pybus et al. (2003), *Genetics* 165(4):2013
  https://pubmed.ncbi.nlm.nih.gov/12644558/
- **HMC for Skygrid**: Baele, Gill, Lemey & Suchard (2020), *Wellcome Open Research*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7463299/
- **HSMRF extension**: Faulkner et al. (2020), *Biometrics* 76:677
  https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13276

## Other implementations to consult
- **phylodyn** (R, BNPR/INLA, fixed tree): https://github.com/mdkarcher/phylodyn
  Good reference for the coalescent likelihood formulation in a non-BEAST context.
- **mlesky** (R, ML skygrid, fixed tree): https://github.com/emvolz-phylodynamics/mlesky
- **phylostan** (Python+Stan, Skyride+Skygrid): https://github.com/4ment/phylostan
- **RevBayes GMRF tutorial**: https://revbayes.github.io/tutorials/coalescent/GMRF

## Existing dl4bi tutorial context
There is already a working DeepRV tutorial in this repo at
`tutorials/deeprv_nonneg_bivariate.ipynb` (branch
`claude/deeprv-nonnegative-tutorial-QQCvh`) demonstrating DeepRV on a bivariate
Normal with softplus non-negativity constraint. The Skygrid tutorial will
eventually follow the same pattern: train DeepRV on the GMRF prior, then use it
as a surrogate inside the NumPyro coalescent model.

## Immediate next steps
1. Fork https://github.com/4ment/torchtree
2. Find and read `PiecewiseConstantCoalescentGridModel` in torchtree source
3. Extract / parse the HCV Egypt tree from `torchtree-experiments`
4. Write the coalescent likelihood in NumPyro/JAX (fixed tree, piecewise constant Nₑ)
5. Write the GMRF prior in NumPyro
6. Run NUTS and compare posterior Nₑ(t) curve against torchtree / BEAST results
