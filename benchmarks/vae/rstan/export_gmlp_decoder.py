"""Train gMLPDeepRV(num_blks=2) on Matern-1/2 GP samples and dump for RStan.

Same idea as export_mlp_decoder.py but for the architecture the DeepRV
paper actually uses. The Stan port in gmlp_decode.stan ties the SGU
LayerNorm params across blocks (Flax shares the default `nn.LayerNorm()`
instance), so this script pulls those params from block 0 only.
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

from dl4bi.vae import gMLPDeepRV


HERE = Path(__file__).parent
OUT = HERE / "gmlp_decoder_artifact.json"

GRID_SIDE = 16
L = GRID_SIDE * GRID_SIDE
COND_DIM = 1
NUM_BLKS = 2
LS_MIN, LS_MAX = 1.0, 100.0
TRAIN_STEPS = 100_000
BATCH = 32
LR = 1e-3
JITTER = 5e-4
N_MATCH_CASES = 4
LOG_EVERY = 10_000


def train_decoder(model, s, rng):
    rng_init, rng_loop = random.split(rng)
    params = model.init(rng_init, jnp.zeros((1, L)), jnp.zeros((COND_DIM,)), s)["params"]

    tx = optax.chain(optax.clip_by_global_norm(3.0),
                     optax.adamw(LR, weight_decay=1e-2))
    opt_state = tx.init(params)
    jitter_eye = JITTER * jnp.eye(L)

    @jax.jit
    def step(params, opt_state, rng):
        rng_ls, rng_z = random.split(rng)
        ls = random.uniform(rng_ls, (), minval=LS_MIN, maxval=LS_MAX)
        z = random.normal(rng_z, (BATCH, L))
        K = matern_1_2(s, s, 1.0, ls) + jitter_eye
        L_chol = jnp.linalg.cholesky(K)
        f = jnp.einsum("ij,bj->bi", L_chol, z)

        def loss_fn(p):
            f_hat = model.apply({"params": p}, z, jnp.array([ls]), s, method="decode")
            return jnp.mean((f_hat.squeeze() - f) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    t0 = time.time()
    loss = jnp.nan
    for i in range(TRAIN_STEPS):
        rng_loop, k = random.split(rng_loop)
        params, opt_state, loss = step(params, opt_state, k)
        if (i + 1) % LOG_EVERY == 0 or i == 0:
            print(f"  step {i + 1:>6d}/{TRAIN_STEPS}: MSE = {float(loss):.4f}")
    print(f"  trained in {time.time() - t0:.1f}s, final MSE = {float(loss):.4f}")
    return params


def arr(x):
    return np.asarray(x, dtype=np.float64).tolist()


def collect_weights(p):
    """Walks the gMLPDeepRV param tree into named tensors that match
    gmlp_decode.stan's `data` block. SGU norm is pulled from block 0 only —
    block 1's SGU uses the same params (Flax param tying)."""
    out = {}
    embed = p["embed"]
    out["embed_W0"] = arr(embed["Dense_0"]["kernel"])
    out["embed_b0"] = arr(embed["Dense_0"]["bias"])
    out["embed_W1"] = arr(embed["Dense_1"]["kernel"])
    out["embed_b1"] = arr(embed["Dense_1"]["bias"])

    g = p["gMLP_0"]
    for k in (0, 1, 2):
        out[f"ln{k}_scale"] = arr(g[f"LayerNorm_{k}"]["scale"])
        out[f"ln{k}_bias"]  = arr(g[f"LayerNorm_{k}"]["bias"])

    sgu_norm = g["gMLPBlock_0"]["SpatialGatingUnit_0"]["norm"]
    out["sgu_norm_scale"] = arr(sgu_norm["scale"])
    out["sgu_norm_bias"]  = arr(sgu_norm["bias"])

    for b in (0, 1):
        blk = g[f"gMLPBlock_{b}"]
        pre = f"b{b}_"
        out[pre + "pi_W0"] = arr(blk["proj_in"]["Dense_0"]["kernel"])
        out[pre + "pi_b0"] = arr(blk["proj_in"]["Dense_0"]["bias"])
        out[pre + "pi_W1"] = arr(blk["proj_in"]["Dense_1"]["kernel"])
        out[pre + "pi_b1"] = arr(blk["proj_in"]["Dense_1"]["bias"])
        sgu = blk["SpatialGatingUnit_0"]
        # (1, L, L) → (L, L); (1, 1, L, 1) → (L,)
        out[pre + "sgu_W"] = arr(np.asarray(sgu["weights"])[0])
        out[pre + "sgu_b"] = arr(np.asarray(sgu["bias"]).reshape(-1))
        out[pre + "po_W0"] = arr(blk["proj_out"]["Dense_0"]["kernel"])
        out[pre + "po_b0"] = arr(blk["proj_out"]["Dense_0"]["bias"])
        out[pre + "po_W1"] = arr(blk["proj_out"]["Dense_1"]["kernel"])
        out[pre + "po_b1"] = arr(blk["proj_out"]["Dense_1"]["bias"])

    head = p["head"]
    out["head_W0"] = arr(head["Dense_0"]["kernel"])
    out["head_b0"] = arr(head["Dense_0"]["bias"])
    out["head_W1"] = arr(head["Dense_1"]["kernel"])
    out["head_b1"] = arr(head["Dense_1"]["bias"])
    return out


def main():
    rng = random.key(0)
    rng_train, rng_match, rng_inf = random.split(rng, 3)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": GRID_SIDE}] * 2).reshape(-1, 2)
    assert s.shape == (L, 2)

    print(f"Training gMLPDeepRV(num_blks={NUM_BLKS}) on Matern-1/2 GP, L={L}, "
          f"ls ~ U({LS_MIN}, {LS_MAX}), {TRAIN_STEPS} steps")
    model = gMLPDeepRV(num_blks=NUM_BLKS)
    params = train_decoder(model, s, rng_train)
    weights = collect_weights(params)

    @jax.jit
    def decode(z, cond):
        return model.apply({"params": params}, z, cond, s, method="decode")

    cases = []
    rng_m = rng_match
    for _ in range(N_MATCH_CASES):
        rng_m, rng_z, rng_c = random.split(rng_m, 3)
        z = random.normal(rng_z, (1, L))
        ls = random.uniform(rng_c, (), minval=LS_MIN, maxval=LS_MAX)
        cond = jnp.array([ls])
        out = decode(z, cond).squeeze()  # (L,)
        cases.append({
            "z": np.asarray(z[0]).tolist(),
            "cond": np.asarray(cond).tolist(),
            "out": np.asarray(out).tolist(),
        })

    rng_mu, rng_y, rng_mask = random.split(rng_inf, 3)
    ls_true, beta_true = 30.0, 1.0
    K_true = matern_1_2(s, s, 1.0, ls_true) + JITTER * jnp.eye(L)
    mu_true = jax.random.multivariate_normal(rng_mu, jnp.zeros(L), K_true)
    rate = jnp.exp(beta_true + mu_true)
    y_obs = jax.random.poisson(rng_y, rate).astype(int)
    mask = (random.uniform(rng_mask, (L,)) < 0.7).astype(int)

    artifact = {
        "arch": "gMLPDeepRV",
        "num_blks": NUM_BLKS,
        "L": L, "cond_dim": COND_DIM,
        "activation": "gelu",
        "input_layout": "cond_as_feats_with_s",
        "training": {"steps": TRAIN_STEPS, "batch": BATCH, "lr": LR,
                     "kernel": "matern_1_2", "ls_prior": [LS_MIN, LS_MAX],
                     "grid_side": GRID_SIDE},
        "s_mat": np.asarray(s).tolist(),
        "weights": weights,
        "test_cases": cases,
        "inference": {
            "y": np.asarray(y_obs).tolist(),
            "obs_mask": np.asarray(mask).tolist(),
            "truth": {"ls": float(ls_true), "beta": float(beta_true),
                      "mu": np.asarray(mu_true).tolist()},
            "source": "true_matern_1_2_GP",
        },
    }
    OUT.write_text(json.dumps(artifact))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:_} bytes)")
    print(f"  {len(cases)} match cases, inference: ls*={ls_true}, beta*={beta_true:+.2f}, "
          f"n_obs={int(mask.sum())}/{L}")


if __name__ == "__main__":
    main()
