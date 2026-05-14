"""Pure-numpy mirror of the Stan `decode` in mlp_decode.stan.

Confirms that the math we will run in Stan reproduces the JAX outputs from
decoder_artifact.json. If this passes, the only remaining risk for the R/Stan
check is the rstan wiring itself, not the model math.
"""

import json
import sys
from pathlib import Path

import numpy as np


def decode(z, cond, layers):
    """One-sample forward; layout matches mlp_decode.stan:decode."""
    x = np.concatenate([z, cond])  # cond_as_locs for a single batch row
    W1, b1 = np.asarray(layers[0]["W"]), np.asarray(layers[0]["b"])
    W2, b2 = np.asarray(layers[1]["W"]), np.asarray(layers[1]["b"])
    h = np.maximum(x @ W1 + b1, 0.0)  # relu
    return h @ W2 + b2


def main():
    artifact = json.loads(Path(__file__).with_name("decoder_artifact.json").read_text())
    layers = artifact["layers"]
    max_diff = 0.0
    for i, case in enumerate(artifact["test_cases"]):
        z = np.asarray(case["z"])
        cond = np.asarray(case["cond"])
        expected = np.asarray(case["out"])
        got = decode(z, cond, layers)
        d = float(np.max(np.abs(got - expected)))
        max_diff = max(max_diff, d)
        print(f"case {i}: max|diff| = {d:.2e}")
    print(f"overall max|diff| = {max_diff:.2e}")
    tol = 1e-5
    if max_diff >= tol:
        print(f"FAIL: exceeds tol {tol}")
        sys.exit(1)
    print(f"PASS (< {tol})")


if __name__ == "__main__":
    main()
