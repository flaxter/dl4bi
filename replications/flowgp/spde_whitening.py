"""SPDE / state-space whitening: an exact, O(m), differentiable GP-prior sampler.

The honest scalable backbone for the linear-Gaussian half of FlowGP (and the
exact alternative to a learned DeepRV). A Matern-nu GP with half-integer nu is a
Gauss-Markov process -- the solution of a linear SDE -- so it has a state-space
form that samples in O(m) by a forward recursion, with no Cholesky. Here we use
Matern-3/2: the state is (f, f'), and on a grid of spacing dt,

    x_{i+1} = A x_i + L_Q eps_i,   x_0 = L_Pinf eps_0,   f_i = (x_i)_0

with A = exp(F dt) closed-form (F has the repeated eigenvalue -lambda,
lambda = sqrt(3)/ell), Pinf = diag(sigma^2, sigma^2 lambda^2), Q = Pinf - A Pinf A^T.

Properties demonstrated:
  * EXACT: empirical covariance matches the dense Matern-3/2 kernel.
  * O(m): the forward recursion scales linearly, vs O(m^3) for the dense Cholesky.
  * DIFFERENTIABLE in the length-scale ell: the whole map eps -> f is smooth in
    ell, so HMC can infer ell with no retraining and no per-ell refactorisation.

This is exactly the amortisation DeepRV learns -- but here it is exact, training-
free, and O(m), for the Markov-kernel case. Conditioning on data is also O(m)
(Kalman forward-filter backward-sample); this script covers the prior whitening,
which is the reparameterisation that DeepRV emulates and that FlowGP whitens with.

Run:
    uv run python replications/flowgp/spde_whitening.py
"""

from __future__ import annotations

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _statespace_matrices(ell, dt, sigma=1.0):
    lam = jnp.sqrt(3.0) / ell
    e = jnp.exp(-lam * dt)
    A = e * jnp.array([[1 + lam * dt, dt], [-(lam**2) * dt, 1 - lam * dt]])
    Pinf = jnp.diag(jnp.array([sigma**2, sigma**2 * lam**2]))
    Q = Pinf - A @ Pinf @ A.T
    L_Q = jnp.linalg.cholesky(Q + 1e-12 * jnp.eye(2))
    L_Pinf = jnp.diag(jnp.array([sigma, sigma * lam]))
    return A, L_Q, L_Pinf


def matern32_sample(eps, ell, sigma=1.0, t0=0.0, t1=1.0):
    """O(m) exact Matern-3/2 sample. eps: (m, 2) ~ N(0, I) -> f: (m,)."""
    m = eps.shape[0]
    dt = (t1 - t0) / (m - 1)
    A, L_Q, L_Pinf = _statespace_matrices(ell, dt, sigma)
    x0 = L_Pinf @ eps[0]

    def step(x, e):
        x = A @ x + L_Q @ e
        return x, x

    _, xs = jax.lax.scan(step, x0, eps[1:])
    return jnp.concatenate([x0[None], xs])[:, 0]


def matern32_kernel(m, ell, sigma=1.0, t0=0.0, t1=1.0):
    t = jnp.linspace(t0, t1, m)
    r = jnp.abs(t[:, None] - t[None, :])
    lam = jnp.sqrt(3.0) / ell
    return sigma**2 * (1 + lam * r) * jnp.exp(-lam * r)


def main():
    key = jax.random.PRNGKey(0)

    # ---- correctness ------------------------------------------------------
    print("[1] Exactness: state-space empirical covariance vs dense kernel")
    m, ell = 80, 0.2
    eps = jax.random.normal(key, (20000, m, 2))
    samp = jax.vmap(matern32_sample, (0, None))(eps, ell)
    emp = jnp.cov(samp.T)
    dense = matern32_kernel(m, ell)
    print(
        f"    max|emp_cov - K| = {float(jnp.max(jnp.abs(emp - dense))):.4f}  "
        f"(Monte-Carlo error; sampler is exact)"
    )

    # ---- scaling: O(m) state-space vs O(m^3) dense Cholesky ---------------
    print("\n[2] Scaling: draw a batch of 64 prior samples")
    print("      m    | dense Cholesky+matvec | state-space O(m)")
    ss = jax.jit(lambda eps, ell: jax.vmap(matern32_sample, (0, None))(eps, ell))
    chol = jax.jit(jnp.linalg.cholesky)
    ms_dense, t_dense, ms_ss, t_ss = [], [], [], []
    for mm in [256, 1024, 4096, 16384, 65536]:
        k = jax.random.PRNGKey(mm)
        e = jax.random.normal(k, (64, mm, 2))
        ss(e, ell).block_until_ready()
        t0 = time.time()
        ss(e, ell).block_until_ready()
        tss = time.time() - t0
        ms_ss.append(mm)
        t_ss.append(tss)
        if mm <= 4096:  # dense O(m^3)/O(m^2 mem) only feasible at small m
            K = matern32_kernel(mm, ell) + 1e-6 * jnp.eye(mm)
            chol(K).block_until_ready()
            t0 = time.time()
            L = chol(K)
            z = jax.random.normal(k, (64, mm))
            (z @ L.T).block_until_ready()
            td = time.time() - t0
            ms_dense.append(mm)
            t_dense.append(td)
            print(f"    {mm:6d} | {td * 1e3:9.1f} ms          | {tss * 1e3:8.1f} ms")
        else:
            print(f"    {mm:6d} | {'(skipped: O(m^3))':>17s}     | {tss * 1e3:8.1f} ms")

    # ---- differentiable in the length-scale (amortisation for free) -------
    print("\n[3] Differentiable in ell (HMC over the kernel needs this):")
    eps0 = jax.random.normal(key, (m, 2))

    def stat(ell):  # any smooth functional of a sample
        return jnp.sum(matern32_sample(eps0, ell) ** 2)

    g = jax.jit(jax.grad(stat))(jnp.array(0.2))
    print(
        f"    d/d(ell) of a sample functional = {float(g):.3f}  (exact, O(m), "
        "no training, no per-ell Cholesky)"
    )

    _plot(ms_dense, t_dense, ms_ss, t_ss, eps[:6], ell)


def _plot(ms_dense, t_dense, ms_ss, t_ss, eps_demo, ell):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(
        ms_dense, t_dense, "o-", color="tab:red", label="dense Cholesky (O(m^3))"
    )
    axes[0].loglog(
        ms_ss, t_ss, "s-", color="tab:green", label="state-space SPDE (O(m))"
    )
    # reference slopes
    mref = jnp.array(ms_ss, dtype=float)
    axes[0].loglog(
        mref, t_ss[0] * (mref / mref[0]), ":", color="gray", lw=1, label="slope 1"
    )
    axes[0].set_xlabel("grid size m")
    axes[0].set_ylabel("time (s)")
    axes[0].set_title("Prior-sample cost: O(m) vs O(m^3)")
    axes[0].legend(fontsize=8)

    t = jnp.linspace(0.0, 1.0, eps_demo.shape[1])
    for e in eps_demo:
        axes[1].plot(t, matern32_sample(e, ell), lw=0.8, alpha=0.7)
    axes[1].set_title("Exact Matern-3/2 samples (O(m) state-space)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")
    fig.tight_layout()
    out = os.path.join(HERE, "spde_whitening.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
