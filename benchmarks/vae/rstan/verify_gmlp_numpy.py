"""Pure-numpy mirror of gmlp_decode.stan:decode. Confirms the Stan math
matches the JAX outputs in gmlp_decoder_artifact.json before R is involved."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr  # standard normal CDF, == Stan's Phi


HERE = Path(__file__).parent


def gelu(x):
    return x * ndtr(x)


def layer_norm(X, gamma, beta_, eps=1e-6):
    m = X.mean(axis=-1, keepdims=True)
    v = X.var(axis=-1, keepdims=True)  # numpy var uses N divisor by default
    return (X - m) / np.sqrt(v + eps) * gamma + beta_


def dense(X, W, b):
    return X @ W + b


def sgu(x, W, b, ln_scale, ln_bias):
    L, D2 = x.shape
    D = D2 // 2
    z1, z2 = x[:, :D], x[:, D:]
    z2 = layer_norm(z2, ln_scale, ln_bias)
    z2 = W @ z2 + b[:, None]
    return z1 * z2


def block(x, w, prefix, sgu_norm_scale, sgu_norm_bias):
    h = gelu(dense(x, w[f"{prefix}pi_W0"], w[f"{prefix}pi_b0"]))
    y = dense(h, w[f"{prefix}pi_W1"], w[f"{prefix}pi_b1"])
    g = sgu(y, w[f"{prefix}sgu_W"], w[f"{prefix}sgu_b"], sgu_norm_scale, sgu_norm_bias)
    u = gelu(dense(g, w[f"{prefix}po_W0"], w[f"{prefix}po_b0"]))
    return dense(u, w[f"{prefix}po_W1"], w[f"{prefix}po_b1"])


def decode(z, cond, s_mat, w):
    L = z.shape[0]
    x = np.concatenate([z[:, None], s_mat, np.broadcast_to(cond, (L, cond.shape[0]))],
                       axis=-1)
    e = gelu(dense(x, w["embed_W0"], w["embed_b0"]))
    e = dense(e, w["embed_W1"], w["embed_b1"])
    for k in (0, 1):
        ln = layer_norm(e, w[f"ln{k}_scale"], w[f"ln{k}_bias"])
        e = e + block(ln, w, f"b{k}_", w["sgu_norm_scale"], w["sgu_norm_bias"])
    e_n = layer_norm(e, w["ln2_scale"], w["ln2_bias"])
    h = gelu(dense(e_n, w["head_W0"], w["head_b0"]))
    out = dense(h, w["head_W1"], w["head_b1"])
    return out[:, 0]


def main():
    artifact = json.loads((HERE / "gmlp_decoder_artifact.json").read_text())
    w = {k: np.asarray(v) for k, v in artifact["weights"].items()}
    s_mat = np.asarray(artifact["s_mat"])
    max_diff = 0.0
    for i, case in enumerate(artifact["test_cases"]):
        z = np.asarray(case["z"])
        cond = np.asarray(case["cond"])
        expected = np.asarray(case["out"])
        got = decode(z, cond, s_mat, w)
        d = float(np.max(np.abs(got - expected)))
        max_diff = max(max_diff, d)
        print(f"case {i}: max|diff| = {d:.2e}")
    print(f"overall max|diff| = {max_diff:.2e}")
    tol = 1e-4   # gMLP has many more ops; float32 noise compounds
    if max_diff >= tol:
        print(f"FAIL: exceeds tol {tol}")
        sys.exit(1)
    print(f"PASS (< {tol})")


if __name__ == "__main__":
    main()
