"""Skygrid coalescent inference on HCV Egypt data using NumPyro NUTS.

Reproduces the classic HCV Egypt analysis (Pybus et al., 2003; Gill et al., 2013)
using the Skygrid model with a GMRF prior on log-population sizes.

Data: 63 HCV E1 sequences sampled in Egypt in 1993.  The input is a fixed
dated phylogeny represented as 62 coalescent-event heights (years before 1993).

Usage:
    python -m dl4bi.coalescent.hcv_egypt [--num-grid N] [--num-samples N]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from dl4bi.coalescent.skygrid import gmrf_log_prob, skygrid_coalescent_log_prob

# ---- HCV Egypt coalescent heights (years before 1993) ----------------------
# 62 internal-node heights from a fixed MCC tree of 63 contemporaneous taxa.
# Source: torchtree/notebooks/coalescent.ipynb

HEIGHTS = jnp.array([
    0.337539776803851, 1.47598666181676, 2.59836642809404,
    10.5419856428014, 10.6808211320844, 15.3216302200854,
    32.9562682261327, 35.2648938177678, 36.7219608418798,
    36.737540711423, 37.2708231729614, 37.7849554073031,
    39.4937183636496, 39.5351866235332, 40.2051118111069,
    40.272699980689, 42.8275583168662, 45.0143000059112,
    46.5121661847954, 47.1448952249262, 47.504526489029,
    47.9616614709545, 48.3715795210146, 48.483790544288,
    48.7943022569899, 50.4087925250658, 51.2112016897755,
    51.2609695712131, 51.878675348271, 54.1904624765934,
    55.4352237750807, 55.5550828199461, 55.9082058859295,
    61.1133486466791, 61.2488301435699, 61.6467848736123,
    61.8450615919513, 63.090453292149, 66.1723097114625,
    66.981821939552, 68.9496036301138, 69.0585863605563,
    69.3140166961911, 70.088816523003, 70.582823656852,
    72.0200679613929, 72.0372853199532, 72.1961653985567,
    72.4862745825324, 74.6256268205132, 74.6985113851702,
    78.4868903769909, 79.3977257023136, 83.4281836229148,
    85.2297867264788, 87.6518990828725, 89.1434858206649,
    94.7629800718393, 224.733954965067, 232.942753855407,
    251.57871299307, 277.961578642117,
])

TAXON_COUNT = len(HEIGHTS) + 1  # 63
SAMPLING_TIMES = jnp.zeros(TAXON_COUNT)  # all sampled in 1993
SAMPLING_YEAR = 1993.0


def make_grid(cutoff: float, num_intervals: int):
    """Create evenly-spaced grid interior points on [0, cutoff].

    Returns (M-1,) array of interior change-points defining M intervals.
    """
    return jnp.linspace(0, cutoff, num_intervals + 1)[1:-1]


def make_gmrf_weights(cutoff: float, num_intervals: int):
    """Grid-spacing weights delta_k for the GMRF prior."""
    grid_full = jnp.linspace(0, cutoff, num_intervals + 1)
    return jnp.diff(grid_full)


def hcv_model(
    node_heights,
    sampling_times,
    grid,
    num_intervals,
):
    """NumPyro model: Skygrid coalescent + GMRF prior."""
    # Precision: tau ~ Gamma(0.001, 0.001)
    precision = numpyro.sample("precision", dist.Gamma(0.001, 0.001))

    # Log population sizes: vague Normal prior (will be overridden by GMRF)
    log_thetas = numpyro.sample(
        "log_thetas",
        dist.Normal(0.0, 10.0).expand([num_intervals]).to_event(1),
    )

    # GMRF smoothing prior (uniform grid => no weights needed)
    gmrf_lp = gmrf_log_prob(log_thetas, precision)
    numpyro.factor("gmrf", gmrf_lp)

    # Coalescent likelihood
    coal_lp = skygrid_coalescent_log_prob(
        log_thetas, grid, node_heights, sampling_times
    )
    numpyro.factor("coalescent", coal_lp)


def run_inference(
    num_grid: int = 75,
    cutoff: float | None = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
):
    """Run NUTS on the HCV Egypt Skygrid model."""
    if cutoff is None:
        cutoff = float(HEIGHTS[-1]) * 1.1  # 10% beyond root

    grid = make_grid(cutoff, num_grid)

    kernel = NUTS(hcv_model, max_tree_depth=10)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        node_heights=HEIGHTS,
        sampling_times=SAMPLING_TIMES,
        grid=grid,
        num_intervals=num_grid,
    )
    mcmc.print_summary()
    return mcmc


def plot_skygrid(mcmc, cutoff, num_grid, output_path="hcv_skygrid.png"):
    """Plot posterior median and 95% CI of N_e(t)."""
    samples = mcmc.get_samples()
    log_thetas = samples["log_thetas"]  # (num_samples, num_grid)

    grid_midpoints = jnp.linspace(0, cutoff, num_grid + 1)
    grid_midpoints = (grid_midpoints[:-1] + grid_midpoints[1:]) / 2.0
    calendar_years = SAMPLING_YEAR - grid_midpoints

    ne_samples = jnp.exp(log_thetas)
    median = jnp.median(ne_samples, axis=0)
    lower = jnp.percentile(ne_samples, 2.5, axis=0)
    upper = jnp.percentile(ne_samples, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(calendar_years, lower, upper, alpha=0.3, label="95% CI")
    ax.plot(calendar_years, median, "k-", linewidth=1.5, label="Median")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Effective population size")
    ax.set_title("HCV Egypt Skygrid (NumPyro NUTS)")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HCV Egypt Skygrid inference")
    parser.add_argument("--num-grid", type=int, default=75)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="hcv_skygrid.png")
    args = parser.parse_args()

    cutoff = args.cutoff
    if cutoff is None:
        cutoff = float(HEIGHTS[-1]) * 1.1

    mcmc = run_inference(
        num_grid=args.num_grid,
        cutoff=cutoff,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
    )
    plot_skygrid(mcmc, cutoff, args.num_grid, args.output)


if __name__ == "__main__":
    main()
