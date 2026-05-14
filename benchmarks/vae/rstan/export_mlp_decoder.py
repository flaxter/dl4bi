"""Train an MLPDeepRV on Matern-1/2 GP samples and dump it for the RStan port.

Pipeline:
  1. Build a small spatial grid (4x4, L=16) — same coordinate system as
     deep_rv_example.py but coarser so the test runs fast.
  2. Train MLPDeepRV(dims=[L, L]) to mimic Cholesky(K_ls) @ z for
     ls ~ Uniform(1, 100), z ~ N(0, I_L). Mirrors
     `dl4bi.vae.train_utils.deep_rv_train_step` but inlined for clarity.
  3. Generate golden (z, cond, decoder_output) cases from the trained model
     so the Stan port can be checked against JAX.
  4. Generate an inference fixture y_obs from the TRUE Matern-1/2 GP at a
     chosen (ls*, beta*) — matches `deep_rv_example.py:gen_y_obs`. The
     Stan model then has a real surrogate-vs-truth recovery task, not
     a self-consistent decoder-vs-decoder one.

Outputs:
  benchmarks/vae/rstan/decoder_artifact.json
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid
from jax import random

from dl4bi.vae import MLPDeepRV


HERE = Path(__file__).parent
OUT = HERE / "decoder_artifact.json"

GRID_SIDE = 4                # 4x4 = 16 locations
L = GRID_SIDE * GRID_SIDE
COND_DIM = 1                 # just ls
DIMS = [L, L]                # MLPDeepRV(dims=[L, L])
LS_MIN, LS_MAX = 1.0, 100.0
TRAIN_STEPS = 20_000
BATCH = 32
LR = 1e-3
JITTER = 5e-4
N_MATCH_CASES = 4


def train_decoder(model, s, rng, steps=TRAIN_STEPS, batch=BATCH, lr=LR):
    """Minimal `deep_rv_train_step` loop inlined to keep this script standalone."""
    rng_init, rng_loop = random.split(rng)
    params = model.init(rng_init, jnp.zeros((1, L)), jnp.zeros((COND_DIM,)))["params"]

    tx = optax.chain(optax.clip_by_global_norm(3.0), optax.adamw(lr, weight_decay=1e-2))
    opt_state = tx.init(params)

    jitter_eye = JITTER * jnp.eye(L)

    @jax.jit
    def step(params, opt_state, rng):
        rng_ls, rng_z = random.split(rng)
        ls = random.uniform(rng_ls, (), minval=LS_MIN, maxval=LS_MAX)
        z = random.normal(rng_z, (batch, L))
        K = matern_1_2(s, s, 1.0, ls) + jitter_eye
        L_chol = jnp.linalg.cholesky(K)
        f = jnp.einsum("ij,bj->bi", L_chol, z)

        def loss_fn(p):
            f_hat = model.apply({"params": p}, z, jnp.array([ls]), method="decode")
            return jnp.mean((f_hat.squeeze() - f) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    t0 = time.time()
    loss = jnp.nan
    for i in range(steps):
        rng_loop, k = random.split(rng_loop)
        params, opt_state, loss = step(params, opt_state, k)
        if (i + 1) % 2_500 == 0 or i == 0:
            print(f"  step {i + 1:>6d}/{steps}: MSE = {float(loss):.4f}")
    print(f"  trained in {time.time() - t0:.1f}s, final MSE = {float(loss):.4f}")
    return params


def main():
    rng = random.key(0)
    rng_train, rng_match, rng_inf = random.split(rng, 3)

    s = build_grid([{"start": 0.0, "stop": 100.0, "num": GRID_SIDE}] * 2).reshape(-1, 2)
    assert s.shape == (L, 2)

    print(f"Training MLPDeepRV(dims={DIMS}) on Matern-1/2 GP, L={L}, "
          f"ls ~ U({LS_MIN}, {LS_MAX}), {TRAIN_STEPS} steps")
    params = train_decoder(MLPDeepRV(dims=DIMS), s, rng_train)

    @jax.jit
    def decode(z, cond):
        return MLPDeepRV(dims=DIMS).apply({"params": params}, z, cond, method="decode")

    inner = params["MLP_0"]
    layers = []
    for name in sorted(inner.keys()):
        layer = inner[name]
        W = np.asarray(layer["kernel"], dtype=np.float64)
        b = np.asarray(layer["bias"], dtype=np.float64)
        layers.append({"name": name, "W": W.tolist(), "b": b.tolist(),
                       "in": int(W.shape[0]), "out": int(W.shape[1])})

    cases = []
    rng_m = rng_match
    for _ in range(N_MATCH_CASES):
        rng_m, rng_z, rng_c = random.split(rng_m, 3)
        z = random.normal(rng_z, (1, L))
        ls = random.uniform(rng_c, (), minval=LS_MIN, maxval=LS_MAX)
        cond = jnp.array([ls])
        out = decode(z, cond)
        cases.append({
            "z": np.asarray(z[0]).tolist(),
            "cond": np.asarray(cond).tolist(),
            "out": np.asarray(out[0]).tolist(),
        })

    # Inference fixture: draw y_obs from the TRUE Matern-1/2 GP, not the
    # surrogate. Mirrors deep_rv_example.py:gen_y_obs.
    rng_mu, rng_y, rng_mask = random.split(rng_inf, 3)
    ls_true, beta_true = 30.0, 1.0  # matches deep_rv_example.py defaults
    K_true = matern_1_2(s, s, 1.0, ls_true) + JITTER * jnp.eye(L)
    mu_true = jax.random.multivariate_normal(rng_mu, jnp.zeros(L), K_true)
    rate = jnp.exp(beta_true + mu_true)
    y_obs = jax.random.poisson(rng_y, rate).astype(int)
    mask = (random.uniform(rng_mask, (L,)) < 0.7).astype(int)

    artifact = {
        "arch": "MLPDeepRV",
        "L": L,
        "cond_dim": COND_DIM,
        "dims": DIMS,
        "activation": "relu",
        "input_layout": "cond_as_locs",
        "training": {
            "steps": TRAIN_STEPS, "batch": BATCH, "lr": LR,
            "kernel": "matern_1_2", "ls_prior": [LS_MIN, LS_MAX],
            "grid_side": GRID_SIDE,
        },
        "s": np.asarray(s).tolist(),
        "layers": layers,
        "test_cases": cases,
        "inference": {
            "y": np.asarray(y_obs).tolist(),
            "obs_mask": np.asarray(mask).tolist(),
            "truth": {"ls": float(ls_true), "beta": float(beta_true),
                      "mu": np.asarray(mu_true).tolist()},
            "source": "true_matern_1_2_GP",
        },
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:_} bytes)")
    print(f"  {len(cases)} match cases, inference: ls*={ls_true}, beta*={beta_true:+.2f}, "
          f"n_obs={int(mask.sum())}/{L}")


if __name__ == "__main__":
    main()
