"""Train the brms.deeprv v0.1 decoder catalog from a YAML config.

For each (grid_size, kernel) combination in the config, train an
`MLPDeepRV(dims=[L, L])` to emulate Cholesky(K_ls) @ z on a 1D unit
interval grid, then dump weights + golden test cases to a per-decoder
JSON. After all decoders finish, write a `manifest.json` mapping each
fingerprint to the JSON's filename.

The JSON is an intermediate format. `pack_decoders.R` converts these
into the canonical `.rds` files described in
`benchmarks/vae/rstan/DESIGN.md` §2.2.

Usage:
    uv run --extra cpu --extra benchmarks python \\
        benchmarks/vae/rstan/train_decoders.py \\
        --config benchmarks/vae/rstan/configs/smoke.yaml

Skips decoders whose JSON already exists with a matching fingerprint;
pass `--force` to retrain unconditionally.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from dl4bi_sps.kernels import matern_1_2, matern_3_2, matern_5_2, rbf
from dl4bi_sps.utils import build_grid
from jax import random

from dl4bi.vae import MLPDeepRV


KERNEL_FNS = {
    "matern_1_2": matern_1_2,
    "matern_3_2": matern_3_2,
    "matern_5_2": matern_5_2,
    "rbf": rbf,
}
SCHEMA_VERSION = 1
SUPPORTED_ARCH = "MLPDeepRV"
SUPPORTED_DOMAIN = "unit_interval"


def fingerprint(arch, arch_version, domain, grid_size, kernel, ls_lo, ls_hi):
    """Stable sha256 of the decoder identity.

    Floats are formatted with %.8f (8 decimal places) — this gives the
    same bytes in Python and R, sidestepping repr() inconsistencies
    between the two languages. ls bounds with more than 8 fractional
    digits of meaningful precision are out of scope for v0.1.
    """
    canonical = "|".join([
        arch, arch_version, domain, str(int(grid_size)), kernel,
        f"{float(ls_lo):.8f}", f"{float(ls_hi):.8f}",
    ])
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_step_fn(model, kfn, s, L, ls_lo, ls_hi, batch, jitter, tx):
    jitter_eye = jitter * jnp.eye(L)

    @jax.jit
    def step(params, opt_state, rng):
        rng_ls, rng_z = random.split(rng)
        ls = random.uniform(rng_ls, (), minval=ls_lo, maxval=ls_hi)
        z = random.normal(rng_z, (batch, L))
        K = kfn(s, s, 1.0, ls) + jitter_eye
        L_chol = jnp.linalg.cholesky(K)
        f = jnp.einsum("ij,bj->bi", L_chol, z)

        def loss_fn(p):
            f_hat = model.apply({"params": p}, z, jnp.array([ls]), method="decode")
            return jnp.mean((f_hat.squeeze() - f) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def train_decoder(grid_size, kernel, cfg, rng):
    L = int(grid_size)
    s = build_grid([{"start": 0.0, "stop": 1.0, "num": L}]).reshape(-1, 1).astype(jnp.float32)
    assert s.shape == (L, 1)

    ls_lo, ls_hi = cfg["ls_trained_range"]
    steps = int(cfg["training"]["steps"])
    batch = int(cfg["training"]["batch"])
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 1e-2))
    clip = float(cfg["training"].get("clip_global_norm", 3.0))
    jitter = float(cfg["training"].get("jitter", 5e-4))

    model = MLPDeepRV(dims=[L, L])
    rng_init, rng_loop = random.split(rng)
    params = model.init(rng_init, jnp.zeros((1, L)), jnp.zeros((1,)))["params"]

    tx = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(lr, weight_decay=wd))
    opt_state = tx.init(params)

    step = make_step_fn(model, KERNEL_FNS[kernel], s, L, ls_lo, ls_hi, batch, jitter, tx)
    log_every = max(steps // 10, 1)

    t0 = time.time()
    loss = jnp.nan
    for i in range(steps):
        rng_loop, k = random.split(rng_loop)
        params, opt_state, loss = step(params, opt_state, k)
        if (i + 1) % log_every == 0 or i == 0:
            print(f"    step {i + 1:>7d}/{steps}: MSE = {float(loss):.4f}")
    elapsed = time.time() - t0
    return params, float(loss), elapsed, s


def build_artifact(grid_size, kernel, cfg, params, final_mse, elapsed, s, rng):
    L = int(grid_size)
    arch_version = cfg["arch_version"]
    ls_lo, ls_hi = cfg["ls_trained_range"]

    inner = params["MLP_0"]
    layer_names = sorted(inner.keys())
    assert len(layer_names) == 2, f"expected 2 MLP layers, got {layer_names}"
    layers = []
    for name in layer_names:
        layer = inner[name]
        W = np.asarray(layer["kernel"], dtype=np.float64)
        b = np.asarray(layer["bias"], dtype=np.float64)
        layers.append({"name": name, "W": W.tolist(), "b": b.tolist()})

    model = MLPDeepRV(dims=[L, L])

    @jax.jit
    def decode(z, cond):
        return model.apply({"params": params}, z, cond, method="decode")

    n_match = int(cfg.get("n_match_cases", 4))
    cases = []
    rng_m = rng
    for _ in range(n_match):
        rng_m, rng_z, rng_c = random.split(rng_m, 3)
        z = random.normal(rng_z, (1, L))
        ls = random.uniform(rng_c, (), minval=ls_lo, maxval=ls_hi)
        cond = jnp.array([ls])
        out = decode(z, cond)
        cases.append({
            "z": np.asarray(z[0], dtype=np.float64).tolist(),
            "cond": np.asarray(cond, dtype=np.float64).tolist(),
            "out": np.asarray(out[0], dtype=np.float64).tolist(),
        })

    fp = fingerprint(
        SUPPORTED_ARCH, arch_version, SUPPORTED_DOMAIN, L, kernel, ls_lo, ls_hi,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "arch": SUPPORTED_ARCH,
        "arch_version": arch_version,
        "domain": SUPPORTED_DOMAIN,
        "grid_size": L,
        "L": L,
        "grid_coords": np.asarray(s[:, 0], dtype=np.float64).tolist(),
        "kernel": kernel,
        "conditionals": ["ls"],
        "ls_trained_range": [float(ls_lo), float(ls_hi)],
        "weights": {
            "W1": layers[0]["W"], "b1": layers[0]["b"],
            "W2": layers[1]["W"], "b2": layers[1]["b"],
        },
        "training": {
            "steps": int(cfg["training"]["steps"]),
            "batch": int(cfg["training"]["batch"]),
            "lr": float(cfg["training"]["lr"]),
            "optimizer": str(cfg["training"].get("optimizer", "adamw")),
            "weight_decay": float(cfg["training"].get("weight_decay", 1e-2)),
            "clip_global_norm": float(cfg["training"].get("clip_global_norm", 3.0)),
            "jitter": float(cfg["training"].get("jitter", 5e-4)),
            "final_mse": final_mse,
            "seed": int(cfg["training"].get("seed", 0)),
            "wall_time_sec": elapsed,
        },
        "test_cases": cases,
        "fingerprint": fp,
    }, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if a matching artifact exists")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    arch = cfg["arch"]
    if arch != SUPPORTED_ARCH:
        raise ValueError(f"arch={arch!r} not supported (v0.1 is {SUPPORTED_ARCH} only)")
    domain = cfg.get("domain", SUPPORTED_DOMAIN)
    if domain != SUPPORTED_DOMAIN:
        raise ValueError(f"domain={domain!r} not supported (v0.1 is {SUPPORTED_DOMAIN} only)")

    unknown_kernels = set(cfg["kernels"]) - set(KERNEL_FNS.keys())
    if unknown_kernels:
        raise ValueError(f"unknown kernels: {sorted(unknown_kernels)}; "
                         f"supported: {sorted(KERNEL_FNS.keys())}")

    seed = int(cfg["training"].get("seed", 0))
    rng_root = random.key(seed)
    arch_version = cfg["arch_version"]
    ls_lo, ls_hi = cfg["ls_trained_range"]

    manifest_entries = {}
    n = len(cfg["grids"]) * len(cfg["kernels"])
    i = 0
    for grid_size in cfg["grids"]:
        for kernel in cfg["kernels"]:
            i += 1
            fp = fingerprint(arch, arch_version, domain, grid_size, kernel, ls_lo, ls_hi)
            stem = f"{domain}_{int(grid_size)}_{kernel}"
            json_path = out_dir / f"{stem}.json"

            if json_path.exists() and not args.force:
                try:
                    existing = json.loads(json_path.read_text())
                except Exception:
                    existing = {}
                if existing.get("fingerprint") == fp:
                    print(f"[{i}/{n}] {json_path.name}: fingerprint match, skipping")
                    manifest_entries[fp] = json_path.name
                    continue
                print(f"[{i}/{n}] {json_path.name}: stale fingerprint, retraining")

            print(f"[{i}/{n}] training {arch} L={grid_size} kernel={kernel} "
                  f"steps={cfg['training']['steps']}")
            rng_root, rng_train, rng_match = random.split(rng_root, 3)
            params, final_mse, elapsed, s = train_decoder(grid_size, kernel, cfg, rng_train)
            artifact, fp = build_artifact(
                grid_size, kernel, cfg, params, final_mse, elapsed, s, rng_match,
            )
            json_path.write_text(json.dumps(artifact))
            print(f"    -> {json_path.name} "
                  f"({json_path.stat().st_size:_} bytes, MSE={final_mse:.4f}, "
                  f"{elapsed:.1f}s)")
            manifest_entries[fp] = json_path.name

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "arch": arch,
        "arch_version": arch_version,
        "domain": domain,
        "ls_trained_range": [float(ls_lo), float(ls_hi)],
        "decoders": manifest_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path} ({len(manifest_entries)} decoders)")


if __name__ == "__main__":
    main()
