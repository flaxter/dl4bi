"""Initialize an MLPDeepRV, dump weights + golden (input, output) pairs to JSON.

The JSON is consumed by:
  - verify_numpy.py (pure-numpy mirror of the Stan forward, runs without R)
  - check_match.R   (rstan; calls the Stan `decode` via expose_stan_functions)

No training is needed for the match test: we only need the same parameters on
both sides. Swap in real trained params later by loading from an orbax
checkpoint and walking `state.params` the same way.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dl4bi.vae import MLPDeepRV


HERE = Path(__file__).parent
OUT = HERE / "decoder_artifact.json"

L = 16
COND_DIM = 1
DIMS = [L, L]
N_TEST_CASES = 4


def main():
    model = MLPDeepRV(dims=DIMS)
    rng = random.key(0)
    z_init = jnp.zeros((1, L))
    cond_init = jnp.zeros((COND_DIM,))
    params = model.init(rng, z_init, cond_init)["params"]

    # Params are nested: {"MLP_0": {"Dense_0": {"kernel", "bias"}, "Dense_1": {...}}}.
    inner = params["MLP_0"]
    layers = []
    for name in sorted(inner.keys()):  # Dense_0, Dense_1 in declared order
        layer = inner[name]
        W = np.asarray(layer["kernel"], dtype=np.float64)
        b = np.asarray(layer["bias"], dtype=np.float64)
        layers.append({"name": name, "W": W.tolist(), "b": b.tolist(),
                       "in": int(W.shape[0]), "out": int(W.shape[1])})

    # Golden test cases: run JAX decode at a handful of (z, cond) inputs.
    @jax.jit
    def decode(z, cond):
        return model.apply({"params": params}, z, cond, method="decode")

    rng_test = random.key(42)
    cases = []
    for i in range(N_TEST_CASES):
        rng_test, rng_z, rng_c = random.split(rng_test, 3)
        z = random.normal(rng_z, (1, L))
        ls = random.uniform(rng_c, (), minval=1.0, maxval=100.0)
        cond = jnp.array([ls])
        out = decode(z, cond)  # shape (1, L)
        cases.append({
            "z": np.asarray(z[0]).tolist(),
            "cond": np.asarray(cond).tolist(),
            "out": np.asarray(out[0]).tolist(),
        })

    # Inference fixture: synthesize Poisson observations from the same decoder
    # at a chosen truth, so HMC has something self-consistent to recover.
    # Matches the numpyro model in deep_rv_example.py:114 (Poisson(exp(beta + mu))
    # with a binary obs mask).
    rng_inf = random.key(123)
    rng_z, rng_ls, rng_beta, rng_mask, rng_y = random.split(rng_inf, 5)
    z_true = random.normal(rng_z, (1, L))
    ls_true = random.uniform(rng_ls, (), minval=1.0, maxval=100.0)
    beta_true = random.normal(rng_beta) * 1.0  # N(0, 1)
    mu_true = decode(z_true, jnp.array([ls_true]))[0]
    rate = jnp.exp(beta_true + mu_true)
    y_obs = jax.random.poisson(rng_y, rate).astype(int)
    mask = (random.uniform(rng_mask, (L,)) < 0.7).astype(int)

    artifact = {
        "arch": "MLPDeepRV",
        "L": L,
        "cond_dim": COND_DIM,
        "dims": DIMS,
        "activation": "relu",
        "input_layout": "cond_as_locs",  # concat([z, cond])
        "layers": layers,
        "test_cases": cases,
        "inference": {
            "y": np.asarray(y_obs).tolist(),
            "obs_mask": np.asarray(mask).tolist(),
            "truth": {
                "z": np.asarray(z_true[0]).tolist(),
                "ls": float(ls_true),
                "beta": float(beta_true),
            },
        },
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(cases)} test cases, L={L})")
    print(f"  inference fixture: ls*={float(ls_true):.3f}, beta*={float(beta_true):+.3f}, "
          f"n_obs={int(mask.sum())}/{L}")


if __name__ == "__main__":
    main()
