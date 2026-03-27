"""Matérn GP Prior for Effective Population Size with DeepRV.

Skygrid coalescent on HCV Egypt (M=75, CUTOFF=400 yr).
Compares GMRF baseline vs Matérn 3/2 GP prior via DeepRV surrogate.

Usage
-----
# Full run (train + MCMC + plot):
    python deeprv_skygrid_matern.py

# Skip training (reuse saved surrogate):
    python deeprv_skygrid_matern.py --skip-training

# Skip training and MCMC (reuse both):
    python deeprv_skygrid_matern.py --skip-training --skip-mcmc

# Longer MCMC with more chains:
    python deeprv_skygrid_matern.py --skip-training --num-chains 4 --num-samples 400 --num-warmup 400
"""

import argparse
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

# ── make dl4bi importable when run from tutorials/ ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io, contextlib
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import wandb
from jax import jit, random
from matplotlib.lines import Line2D
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from dl4bi.coalescent.skygrid import gmrf_log_prob, skygrid_coalescent_log_prob
from dl4bi.coalescent.hcv_egypt import HEIGHTS, SAMPLING_TIMES
from dl4bi.core.train import Callback, cosine_annealing_lr, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder

wandb.init(mode="disabled")

# ── constants ─────────────────────────────────────────────────────────────────
M            = 75
CUTOFF       = 400.0
SEED         = 42
SAMPLING_YEAR = 1993.0

grid_full  = jnp.linspace(0, CUTOFF, M + 1)
grid       = grid_full[1:-1]
grid_times = ((grid_full[:-1] + grid_full[1:]) / 2)[:, None]   # (M, 1)
times_flat = grid_times[:, 0]                                    # (M,)
s          = grid_times

# GP hyperparameter priors (shared by training dataloader and MCMC model)
LOG_SIGMA2_MEAN = 2.71;  LOG_SIGMA2_STD = 0.5   # sigma2 95% CI ~ [5.5, 40], mode ~15
LOG_ELL_MEAN    = float(jnp.log(CUTOFF / 4));  LOG_ELL_STD = 0.3   # ell 95% CI ~ [55, 182] yr


# ── Matérn 3/2 kernel ─────────────────────────────────────────────────────────
@jit
def matern32_kernel(times, log_sigma2, log_ell, jitter=1e-3):
    """jitter=1e-3 keeps condition number ≤ M/jitter ~ 75k (safe in float32)."""
    sigma2 = jnp.exp(log_sigma2)
    ell    = jnp.exp(log_ell)
    d = jnp.abs(times[:, None] - times[None, :])
    r = jnp.sqrt(3.0) * d / ell
    K = sigma2 * (1.0 + r) * jnp.exp(-r)
    return K + jitter * jnp.eye(times.shape[0])


@jit
def gp_cholesky(times, log_sigma2, log_ell):
    return jnp.linalg.cholesky(matern32_kernel(times, log_sigma2, log_ell))


# ── training dataloader ───────────────────────────────────────────────────────
def gen_train_dataloader(batch_size=64):
    def dataloader(rng):
        while True:
            rng, rng_s2, rng_ell, rng_z = random.split(rng, 4)
            log_sigma2 = dist.Normal(LOG_SIGMA2_MEAN, LOG_SIGMA2_STD).sample(rng_s2)
            log_ell    = dist.Normal(LOG_ELL_MEAN,    LOG_ELL_STD).sample(rng_ell)
            L = gp_cholesky(times_flat, log_sigma2, log_ell)
            z = dist.Normal().sample(rng_z, (batch_size, M))
            f = jnp.einsum("ij,bj->bi", L, z)
            yield {
                "s":            s,
                "z":            z,
                "conditionals": jnp.array([log_sigma2, log_ell]),
                "f":            f,
            }
    return dataloader


# ── numpyro models ────────────────────────────────────────────────────────────
def gmrf_model(node_heights, sampling_times, grid):
    precision  = numpyro.sample("precision", dist.Gamma(0.001, 0.001))
    log_thetas = numpyro.sample("log_thetas",
                                dist.Normal(0.0, 10.0).expand([M]).to_event(1))
    numpyro.factor("gmrf", gmrf_log_prob(log_thetas, precision))
    numpyro.factor("coalescent",
        skygrid_coalescent_log_prob(log_thetas, grid, node_heights, sampling_times))


def matern_deeprv_model(surrogate_decoder, node_heights, sampling_times, grid):
    log_sigma2 = numpyro.sample("log_sigma2",
                                dist.Normal(LOG_SIGMA2_MEAN, LOG_SIGMA2_STD))
    log_ell    = numpyro.sample("log_ell",
                                dist.Normal(LOG_ELL_MEAN, LOG_ELL_STD))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
    conditionals = jnp.array([log_sigma2, log_ell])
    log_thetas = numpyro.deterministic(
        "log_thetas",
        surrogate_decoder(z[None], conditionals, s=s)[0, :, 0],
    )
    numpyro.factor("coalescent",
        skygrid_coalescent_log_prob(log_thetas, grid, node_heights, sampling_times))


# ── phases ────────────────────────────────────────────────────────────────────
def phase_train(args):
    print(f"\n{'='*60}\n  Training DeepRV surrogate ({args.n_steps:,} steps)\n{'='*60}")
    rng = random.key(SEED)
    rng_train, rng_vis = random.split(rng)

    rng_cb     = random.fold_in(rng_vis, 101)
    Z_CB       = dist.Normal().sample(rng_cb, (50, M))
    LS2_CB     = jnp.log(jnp.array(2.0))
    LELL_CB    = jnp.log(jnp.array(CUTOFF / 4))
    L_CB       = gp_cholesky(times_flat, LS2_CB, LELL_CB)
    F_CB_EXACT = jnp.einsum("ij,bj->bi", L_CB, Z_CB)

    curve_steps, curve_rmse, snapshots = [], [], {}
    snapshot_steps = {400, 2_000, 5_000, args.n_steps}
    nn_model = gMLPDeepRV(num_blks=3)

    def record_progress(step, rng_step, state, batch, extra):
        f_hat = nn_model.apply(
            {"params": state.params, **state.kwargs},
            Z_CB, jnp.array([LS2_CB, LELL_CB]), s=s,
            rngs={"extra": rng_step},
        ).f_hat.squeeze(-1)
        rmse = float(jnp.sqrt(jnp.mean((f_hat - F_CB_EXACT) ** 2)))
        curve_steps.append(step)
        curve_rmse.append(rmse)
        if step in snapshot_steps:
            snapshots[step] = (state.params, state.kwargs)
            print(f"  step {step:>6,}  RMSE = {rmse:.4f}")

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(args.n_steps, 1e-3), weight_decay=1e-2),
    )

    state = train(
        rng_train, nn_model, optimizer, deep_rv_train_step,
        args.n_steps, gen_train_dataloader(),
        callbacks=[Callback(fn=record_progress, interval=200)],
    )

    surrogate_decoder = generate_surrogate_decoder(state, nn_model)

    # Save
    with open(args.surrogate_path, "wb") as f:
        pickle.dump({
            "params": state.params,
            "kwargs": state.kwargs,
            "curve_steps": curve_steps,
            "curve_rmse":  curve_rmse,
            "snapshots":   snapshots,
            "Z_CB": np.array(Z_CB),
            "F_CB_EXACT": np.array(F_CB_EXACT),
            "LS2_CB": float(LS2_CB),
            "LELL_CB": float(LELL_CB),
        }, f)
    print(f"  Surrogate saved → {args.surrogate_path}")

    _plot_training_curve(curve_steps, curve_rmse, snapshots, Z_CB, F_CB_EXACT,
                         LS2_CB, LELL_CB, nn_model, rng_vis)
    return surrogate_decoder


def phase_load_surrogate(args):
    print(f"  Loading surrogate from {args.surrogate_path}")
    with open(args.surrogate_path, "rb") as f:
        ckpt = pickle.load(f)
    nn_model = gMLPDeepRV(num_blks=3)

    _plot_training_curve(
        ckpt["curve_steps"], ckpt["curve_rmse"], ckpt["snapshots"],
        jnp.array(ckpt["Z_CB"]), jnp.array(ckpt["F_CB_EXACT"]),
        ckpt["LS2_CB"], ckpt["LELL_CB"], nn_model,
        random.key(SEED + 1),
    )

    from dl4bi.core.train import TrainState
    import flax.linen as nn_flax
    # reconstruct a minimal TrainState-like object that generate_surrogate_decoder needs
    class _FakeState:
        params = ckpt["params"]
        kwargs = ckpt["kwargs"]
    return generate_surrogate_decoder(_FakeState(), nn_model)


def phase_mcmc(args, surrogate_decoder):
    rng = random.key(SEED + 10)
    rng_gmrf, rng_drv = random.split(rng)

    common = dict(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
        progress_bar=True,
    )

    print(f"\n{'='*60}\n  GMRF baseline  ({args.num_chains} chains × {args.num_samples} samples)\n{'='*60}")
    mcmc_gmrf = MCMC(NUTS(gmrf_model, max_tree_depth=10), **common)
    mcmc_gmrf.run(rng_gmrf, node_heights=HEIGHTS, sampling_times=SAMPLING_TIMES, grid=grid)
    mcmc_gmrf.print_summary(exclude_deterministic=True)

    print(f"\n{'='*60}\n  Matérn 3/2 GP + DeepRV  ({args.num_chains} chains × {args.num_samples} samples)\n{'='*60}")
    mcmc_drv = MCMC(NUTS(matern_deeprv_model, max_tree_depth=10), **common)
    mcmc_drv.run(rng_drv, surrogate_decoder=surrogate_decoder,
                 node_heights=HEIGHTS, sampling_times=SAMPLING_TIMES, grid=grid)
    mcmc_drv.print_summary(exclude_deterministic=True)

    with open(args.mcmc_path, "wb") as f:
        pickle.dump({"gmrf": mcmc_gmrf.get_samples(),
                     "drv":  mcmc_drv.get_samples()}, f)
    print(f"  MCMC samples saved → {args.mcmc_path}")
    return mcmc_gmrf, mcmc_drv


def phase_load_mcmc(args):
    print(f"  Loading MCMC samples from {args.mcmc_path}")
    with open(args.mcmc_path, "rb") as f:
        d = pickle.load(f)

    class _FakeMCMC:
        def __init__(self, samples): self._samples = samples
        def get_samples(self): return self._samples

    return _FakeMCMC(d["gmrf"]), _FakeMCMC(d["drv"])


def phase_plot(mcmc_gmrf, mcmc_drv, out_path="hcv_egypt_ne.png"):
    t_axis = np.array(times_flat)

    def plot_ne(ax, mcmc, label, color):
        log_thetas = mcmc.get_samples()["log_thetas"]
        # handle (chains, samples, M) or (samples, M)
        if log_thetas.ndim == 3:
            log_thetas = log_thetas.reshape(-1, M)
        ne     = np.exp(np.array(log_thetas))
        median = np.median(ne, axis=0)
        lo     = np.percentile(ne, 2.5,  axis=0)
        hi     = np.percentile(ne, 97.5, axis=0)
        cal    = SAMPLING_YEAR - t_axis
        ax.fill_between(cal, lo, hi, alpha=0.25, color=color)
        ax.plot(cal, median, color=color, lw=2, label=label)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_ne(ax, mcmc_gmrf, "GMRF (baseline)",          color="steelblue")
    plot_ne(ax, mcmc_drv,  "Matérn 3/2 GP + DeepRV",  color="tomato")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Calendar year")
    ax.set_ylabel(r"Effective population size  $N_e(t)$")
    ax.set_title(r"HCV Egypt: posterior $N_e(t)$  --  GMRF vs Matérn GP prior")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved → {out_path}")
    plt.show()

    # ── GP hyperparameter posteriors ─────────────────────────────────────────
    drv_samples = mcmc_drv.get_samples()
    if "log_sigma2" in drv_samples and "log_ell" in drv_samples:
        log_s2  = np.array(drv_samples["log_sigma2"]).ravel()
        log_ell = np.array(drv_samples["log_ell"]).ravel()
        sigma2  = np.exp(log_s2)
        ell     = np.exp(log_ell)

        # prior densities for reference
        from scipy.stats import norm
        s2_grid  = np.linspace(sigma2.min() * 0.8, sigma2.max() * 1.2, 300)
        ell_grid = np.linspace(ell.min() * 0.8, ell.max() * 1.2, 300)
        prior_s2  = norm.pdf(np.log(s2_grid), LOG_SIGMA2_MEAN, LOG_SIGMA2_STD) / s2_grid
        prior_ell = norm.pdf(np.log(ell_grid), LOG_ELL_MEAN, LOG_ELL_STD) / ell_grid

        hp_path = out_path.replace(".png", "_hyperparams.png")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(sigma2, bins=60, density=True, color="tomato", alpha=0.5,
                     label="Posterior")
        axes[0].plot(s2_grid, prior_s2, "k--", lw=1.5, label="Prior")
        axes[0].set_xlabel(r"Signal variance  $\sigma^2$")
        axes[0].set_ylabel("Density")
        axes[0].set_title(r"Posterior  $\sigma^2$")
        axes[0].legend()
        axes[0].grid(True, alpha=0.2)

        axes[1].hist(ell, bins=60, density=True, color="tomato", alpha=0.5,
                     label="Posterior")
        axes[1].plot(ell_grid, prior_ell, "k--", lw=1.5, label="Prior")
        axes[1].set_xlabel(r"Lengthscale  $\ell$  (years)")
        axes[1].set_ylabel("Density")
        axes[1].set_title(r"Posterior  $\ell$")
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)

        fig.suptitle(r"Matérn 3/2 GP hyperparameter posteriors", fontsize=13)
        fig.tight_layout()
        plt.savefig(hp_path, dpi=150)
        print(f"  Plot saved → {hp_path}")
        plt.show()


# ── internal: training curve plot ─────────────────────────────────────────────
def _plot_training_curve(curve_steps, curve_rmse, snapshots,
                         Z_CB, F_CB_EXACT, LS2_CB, LELL_CB,
                         nn_model, rng_vis):
    snap_steps_sorted = sorted(snapshots.keys())
    if not snap_steps_sorted:
        print("  (no snapshots to plot)")
        return

    n_traj = 8
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, len(snap_steps_sorted), hspace=0.45, wspace=0.28)

    ax_c = fig.add_subplot(gs[0, :])
    finite = [(step, r) for step, r in zip(curve_steps, curve_rmse)
              if np.isfinite(r) and r > 0]
    if finite:
        px, py = zip(*finite)
        ax_c.plot(px, py, color="steelblue", lw=1.5)
        ax_c.set_yscale("log")
        rmse_max = max(py)
    else:
        rmse_max = 1.0
        print("  WARNING: all RMSE values are NaN — training may have failed")
    for step in snap_steps_sorted:
        ax_c.axvline(step, color="crimson", ls="--", lw=1, alpha=0.7)
        ax_c.text(step, rmse_max * 0.97, f" {step:,}", color="crimson",
                  fontsize=8, va="top")
    ax_c.set_xlabel("Training step")
    ax_c.set_ylabel(r"RMSE  (held-out, $\sigma^2$=2, $\ell$=cutoff/4)")
    ax_c.set_title("DeepRV training curve")
    ax_c.grid(True, which="both", alpha=0.3)

    t_axis = np.array(times_flat)
    for col, step in enumerate(snap_steps_sorted):
        ax = fig.add_subplot(gs[1, col])
        params, kwargs = snapshots[step]
        f_hat_snap = nn_model.apply(
            {"params": params, **kwargs},
            Z_CB[:n_traj], jnp.array([LS2_CB, LELL_CB]), s=s,
            rngs={"extra": random.fold_in(rng_vis, step)},
        ).f_hat.squeeze(-1)
        for i in range(n_traj):
            ax.plot(t_axis, np.array(F_CB_EXACT[i]),      color="steelblue", alpha=0.5, lw=1.2)
            ax.plot(t_axis, np.array(f_hat_snap[i]),      color="tomato",    alpha=0.5, lw=1.2, ls="--")
        ax.set_title(f"Step {step:,}")
        ax.set_xlabel("Years before present")
        if col == 0:
            ax.set_ylabel("log Ne")
        if col == len(snap_steps_sorted) - 1:
            ax.legend([Line2D([0],[0], color="steelblue"),
                       Line2D([0],[0], color="tomato", ls="--")],
                      ["Exact L z", "DeepRV"], fontsize=8, loc="upper right")
    plt.suptitle(r"DeepRV approximation quality  ($\sigma^2$=2, $\ell$=cutoff/4)",
                 fontsize=12, y=1.02)
    plt.savefig("deeprv_training_curve.png", dpi=120, bbox_inches="tight")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-training", action="store_true",
                   help="Load surrogate from --surrogate-path instead of training")
    p.add_argument("--skip-mcmc",     action="store_true",
                   help="Load MCMC samples from --mcmc-path instead of running")
    p.add_argument("--surrogate-path", default="surrogate.pkl")
    p.add_argument("--mcmc-path",      default="mcmc_samples.pkl")
    p.add_argument("--n-steps",   type=int, default=15_000,
                   help="DeepRV training steps (default 15000)")
    p.add_argument("--num-warmup",  type=int, default=400)
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--num-chains",  type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"M={M}, CUTOFF={CUTOFF}, SEED={SEED}")

    # ── 1. surrogate ──────────────────────────────────────────────────────────
    if args.skip_training:
        surrogate_decoder = phase_load_surrogate(args)
    else:
        surrogate_decoder = phase_train(args)

    # ── 2. MCMC ───────────────────────────────────────────────────────────────
    if args.skip_mcmc:
        mcmc_gmrf, mcmc_drv = phase_load_mcmc(args)
    else:
        mcmc_gmrf, mcmc_drv = phase_mcmc(args, surrogate_decoder)

    # ── 3. plot ───────────────────────────────────────────────────────────────
    phase_plot(mcmc_gmrf, mcmc_drv)
