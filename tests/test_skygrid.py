"""Tests for Skygrid coalescent likelihood and GMRF prior.

Validates against reference values from torchtree (Fourment et al., 2024).
"""

import jax.numpy as jnp
import pytest

from dl4bi.coalescent.skygrid import gmrf_log_prob, skygrid_coalescent_log_prob


def test_skygrid_homochronous():
    """4 taxa, contemporaneous, grid on [0,10] with 5 intervals."""
    sampling_times = jnp.array([0.0, 0.0, 0.0, 0.0])
    heights = jnp.array([2.0, 6.0, 12.0])
    thetas = jnp.array([3.0, 10.0, 4.0, 2.0, 3.0])
    log_thetas = jnp.log(thetas)
    grid = jnp.linspace(0, 10.0, 5)[1:]

    log_p = skygrid_coalescent_log_prob(log_thetas, grid, heights, sampling_times)
    assert float(log_p) == pytest.approx(-11.8751856, abs=1e-4)


def test_skygrid_heterochronous_cutoff10():
    """5 taxa, heterochronous, cutoff=10."""
    sampling_times = jnp.array([0.0, 1.0, 2.0, 3.0, 12.0])
    thetas_log = jnp.array([1.0, 3.0, 6.0, 8.0, 9.0])
    heights = jnp.array([1.5, 4.0, 6.0, 16.0])
    grid = jnp.linspace(0, 10.0, 5)[1:]

    log_p = skygrid_coalescent_log_prob(thetas_log, grid, heights, sampling_times)
    assert float(log_p) == pytest.approx(-19.594893640219844, abs=1e-4)


def test_skygrid_heterochronous_cutoff18():
    """5 taxa, heterochronous, cutoff=18."""
    sampling_times = jnp.array([0.0, 1.0, 2.0, 3.0, 12.0])
    thetas_log = jnp.array([1.0, 3.0, 6.0, 8.0, 9.0])
    heights = jnp.array([1.5, 4.0, 6.0, 16.0])
    grid = jnp.linspace(0, 18.0, 5)[1:]

    log_p = skygrid_coalescent_log_prob(thetas_log, grid, heights, sampling_times)
    assert float(log_p) == pytest.approx(-14.918634593243764, abs=1e-4)


def test_gmrf_unit_precision():
    """GMRF with unit precision, no weights."""
    log_thetas = jnp.array([1.0, 2.0, 3.0])
    precision = jnp.array(1.0)
    lp = gmrf_log_prob(log_thetas, precision)
    # dim=2, diff_sq=[1,1], sum=2
    # log_p = 2/2*log(1) - 1/2*2 - 2/2*log(2*pi) = -1 - log(2*pi)
    expected = -1.0 - 1.8378770664093453
    assert float(lp) == pytest.approx(expected, abs=1e-5)


def test_gmrf_with_weights():
    """GMRF with non-unit weights."""
    log_thetas = jnp.array([1.0, 3.0, 4.0])
    precision = jnp.array(2.0)
    weights = jnp.array([2.0, 0.5])
    lp = gmrf_log_prob(log_thetas, precision, weights)
    # diff = [2, 1], diff_sq/weights = [4/2, 1/0.5] = [2, 2], sum = 4
    # dim = 2
    # log_p = 2/2*log(2) - 2/2*4 - 2/2*log(2*pi)
    #       = log(2) - 4 - log(2*pi)
    import math

    expected = math.log(2) - 4.0 - 1.8378770664093453
    assert float(lp) == pytest.approx(expected, abs=1e-5)
