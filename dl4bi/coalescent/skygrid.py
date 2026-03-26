"""Skygrid coalescent model in NumPyro/JAX.

Implements the piecewise-constant coalescent likelihood on a fixed grid
(Gill et al., 2013, MBE 30(3):713) with a GMRF smoothing prior on
log-population sizes, suitable for NUTS inference in NumPyro.

Reference implementation: torchtree (Fourment et al., 2024)
  PiecewiseConstantCoalescentGrid in torchtree/evolution/coalescent.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


# --------------------------------------------------------------------------- #
# Coalescent likelihood
# --------------------------------------------------------------------------- #


def skygrid_coalescent_log_prob(
    log_thetas: jnp.ndarray,
    grid: jnp.ndarray,
    node_heights: jnp.ndarray,
    sampling_times: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability of a fixed tree under the Skygrid coalescent.

    Parameters
    ----------
    log_thetas : (M,) array
        Log effective population sizes on the grid, gamma_k = log(N_e) in each
        of the M grid intervals.
    grid : (M-1,) array
        Interior grid-point times (excluding 0).  E.g. for M=5 intervals on
        [0, cutoff] with equal spacing: ``jnp.linspace(0, cutoff, M+1)[1:-1]``
        gives the M-1 interior points.  NOTE: we need M-1 *interior* change-
        points to define M intervals: [0, g1), [g1, g2), ..., [g_{M-1}, inf).
    node_heights : (n-1,) array
        Internal-node heights (coalescent-event times), going backwards from
        the present.  For n taxa there are n-1 coalescent events.
    sampling_times : (n,) array
        Tip sampling times (0 for contemporaneous data).

    Returns
    -------
    Scalar log-likelihood.
    """
    n_taxa = sampling_times.shape[0]

    # Combine all event times: sampling (+1), coalescent (-1), grid (0)
    all_heights = jnp.concatenate([sampling_times, node_heights, grid])
    node_mask = jnp.concatenate([
        jnp.ones(n_taxa, dtype=jnp.int32),           # sampling
        -jnp.ones(node_heights.shape[0], dtype=jnp.int32),  # coalescent
        jnp.zeros(grid.shape[0], dtype=jnp.int32),   # grid
    ])

    # Sort by height
    indices = jnp.argsort(all_heights)
    heights_sorted = all_heights[indices]
    node_mask_sorted = node_mask[indices]

    # Lineage count via cumulative sum of mask (+1 sample, -1 coalescence)
    lineage_count = jnp.cumsum(node_mask_sorted)[:-1]

    # Durations between consecutive sorted events
    durations = jnp.diff(heights_sorted)

    # C(k,2) = k*(k-1)/2
    lchoose2 = lineage_count * (lineage_count - 1) / 2.0

    # Map each position to its grid-interval index via cumulative count of
    # grid markers (node_mask == 0).
    thetas_indices = jnp.cumsum(node_mask_sorted == 0)  # shape (N,)
    # Clamp to valid range [0, M-1]
    thetas_indices = jnp.clip(thetas_indices, 0, log_thetas.shape[0] - 1)

    log_thetas_sorted = log_thetas[thetas_indices]
    thetas_sorted = jnp.exp(log_thetas_sorted)

    # Only include log(theta) terms at coalescent events
    is_coalescent = node_mask_sorted == -1
    log_theta_coal = jnp.where(is_coalescent, log_thetas_sorted, 0.0)

    # log p = sum_i [ -C(k_i,2) * dt_i / theta_i ] - sum_{coal events} log(theta_j)
    log_lik = jnp.sum(
        -lchoose2 * durations / thetas_sorted[:-1] - log_theta_coal[1:]
    )
    return log_lik


# --------------------------------------------------------------------------- #
# GMRF prior
# --------------------------------------------------------------------------- #


def gmrf_log_prob(
    log_thetas: jnp.ndarray,
    precision: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Log-density of a first-order GMRF (intrinsic) prior.

    .. math::
        p(x | tau) = prod_{i=1}^{M-1}  sqrt(tau / 2pi)
                     exp(-tau/2 (x_{i+1} - x_i)^2 / delta_i)

    Parameters
    ----------
    log_thetas : (M,) array
        Field values (log population sizes).
    precision : scalar
        Precision (inverse variance) parameter tau.
    weights : (M-1,) array or None
        Grid spacings delta_k = s_{k+1} - s_k used to weight the differences.
        If None, unit weights are used (uniform grid spacing).

    Returns
    -------
    Scalar log-density.
    """
    diff = jnp.diff(log_thetas)
    diff_sq = diff ** 2
    if weights is not None:
        diff_sq = diff_sq / weights
    dim = log_thetas.shape[0] - 1.0
    log_2pi = 1.8378770664093453
    return (
        dim / 2.0 * jnp.log(precision)
        - precision / 2.0 * jnp.sum(diff_sq)
        - dim / 2.0 * log_2pi
    )


# --------------------------------------------------------------------------- #
# NumPyro model
# --------------------------------------------------------------------------- #


def skygrid_model(
    node_heights: jnp.ndarray,
    sampling_times: jnp.ndarray,
    grid: jnp.ndarray,
    num_grid_intervals: int,
    gmrf_weights: jnp.ndarray | None = None,
    precision_shape: float = 0.001,
    precision_rate: float = 0.001,
):
    """NumPyro model for Skygrid coalescent with GMRF prior.

    Samples log-population sizes gamma_1..gamma_M and precision tau, then
    scores the GMRF prior and coalescent likelihood as factors.

    Parameters
    ----------
    node_heights : (n-1,) array
        Fixed coalescent-event times.
    sampling_times : (n,) array
        Tip sampling times.
    grid : (M-1,) array
        Interior grid-point times.
    num_grid_intervals : int
        Number of grid intervals M (= len(grid) + 1).
    gmrf_weights : (M-1,) array or None
        Grid spacings for GMRF weighting.
    precision_shape, precision_rate : float
        Gamma prior hyperparameters for the GMRF precision.
    """
    # --- Precision parameter ---
    precision = numpyro.sample(
        "precision", dist.Gamma(precision_shape, precision_rate)
    )

    # --- Log population sizes ---
    log_thetas = numpyro.sample(
        "log_thetas",
        dist.Normal(0.0, 10.0).expand([num_grid_intervals]).to_event(1),
    )

    # --- GMRF prior (added as a factor) ---
    gmrf_lp = gmrf_log_prob(log_thetas, precision, gmrf_weights)
    numpyro.factor("gmrf", gmrf_lp)

    # --- Coalescent likelihood ---
    coal_lp = skygrid_coalescent_log_prob(
        log_thetas, grid, node_heights, sampling_times
    )
    numpyro.factor("coalescent", coal_lp)
