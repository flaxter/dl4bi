"""Reusable GP machinery for the FlowGP physics experiments.

Provides an RBF kernel with per-dimension length-scales, marginal-likelihood
fitting of (length-scales, output-scale, affine mean) with fixed observation
noise, a closed-form posterior, and the kernel-smoothing extension of grid
samples to off-grid test points (paper Appendix H.1, Eq. 65).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def rbf_kernel(xa, xb, log_ls, log_os):
    """k(x, x') = exp(log_os) * exp(-0.5 sum_d ((x_d - x'_d) / ls_d)^2).

    xa: (n, d), xb: (m, d), log_ls: (d,) log length-scales, log_os: scalar.
    """
    ls = jnp.exp(log_ls)
    diff = (xa[:, None, :] - xb[None, :, :]) / ls
    return jnp.exp(log_os) * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))


def _unpack(params, d):
    log_ls = params[:d]
    log_os = params[d]
    a = params[d + 1]
    b = params[d + 2 : 2 * d + 2]
    return log_ls, log_os, a, b


def _neg_log_marglik(params, X, y, noise_var, d, jitter):
    log_ls, log_os, a, b = _unpack(params, d)
    n = X.shape[0]
    K = rbf_kernel(X, X, log_ls, log_os) + (noise_var + jitter) * jnp.eye(n)
    r = y - (a + X @ b)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), r)
    return (
        0.5 * r @ alpha
        + jnp.sum(jnp.log(jnp.diag(L)))
        + 0.5 * n * jnp.log(2.0 * jnp.pi)
    )


def fit_gp(X, y, noise_var, init_log_ls=None, init_log_os=None, jitter=1e-8):
    """Fit length-scales, output-scale, and an affine mean by maximising the
    marginal likelihood (observation noise held fixed). Returns a params dict."""
    d = X.shape[1]
    if init_log_ls is None:
        init_log_ls = np.full(d, np.log(0.1))
    if init_log_os is None:
        init_log_os = float(np.log(np.var(np.asarray(y)) + 1e-6))
    p0 = np.concatenate(
        [np.asarray(init_log_ls, float), [init_log_os, float(np.mean(y))], np.zeros(d)]
    )

    vg = jax.jit(
        jax.value_and_grad(lambda p: _neg_log_marglik(p, X, y, noise_var, d, jitter))
    )

    def scipy_obj(p):
        v, g = vg(jnp.asarray(p))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(scipy_obj, p0, jac=True, method="L-BFGS-B")
    log_ls, log_os, a, b = _unpack(jnp.asarray(res.x), d)
    return {
        "log_ls": log_ls,
        "log_os": log_os,
        "a": a,
        "b": b,
        "noise_var": noise_var,
        "nll": float(res.fun),
    }


def gp_posterior(Xo, yo, params, jitter=1e-8) -> tuple[Callable, Callable]:
    """Closed-form GP posterior given observations and fitted hyperparameters.

    Returns (mean_fn, cov_fn): mean_fn(X) is the posterior mean at X, and
    cov_fn(Xa, Xb) is the posterior cross-covariance between Xa and Xb.
    """
    log_ls, log_os = params["log_ls"], params["log_os"]
    a, b, noise_var = params["a"], params["b"], params["noise_var"]
    n = Xo.shape[0]
    Koo = rbf_kernel(Xo, Xo, log_ls, log_os) + (noise_var + jitter) * jnp.eye(n)
    Lo = jnp.linalg.cholesky(Koo)
    resid = yo - (a + Xo @ b)
    alpha = jax.scipy.linalg.cho_solve((Lo, True), resid)

    def mean(X):
        return a + X @ b + rbf_kernel(X, Xo, log_ls, log_os) @ alpha

    def cov(Xa, Xb):
        Kab = rbf_kernel(Xa, Xb, log_ls, log_os)
        Kao = rbf_kernel(Xa, Xo, log_ls, log_os)
        Kob = rbf_kernel(Xo, Xb, log_ls, log_os)
        return Kab - Kao @ jax.scipy.linalg.cho_solve((Lo, True), Kob)

    return mean, cov


def extend_to_test(
    samples, X_grid, X_test, m_pred, K_pred, mean_fn, cov_fn, jitter=1e-8
):
    """Map grid samples to off-grid test points (Eq. 65).

    f_hat(x*) = mu_{*|y}(x*) + k_{*|y}(x*, X_grid) K_{**|y}^{-1} (f - m_pred).
    """
    m = X_grid.shape[0]
    K = K_pred + jitter * jnp.eye(m)
    L = jnp.linalg.cholesky(K)
    weights = jax.scipy.linalg.cho_solve((L, True), (samples - m_pred).T)  # (m, S)
    cross = cov_fn(X_test, X_grid)  # (n_test, m) posterior cross-cov
    return mean_fn(X_test)[None, :] + (cross @ weights).T  # (S, n_test)


def predictive_metrics(test_samples, y_test, noise_var):
    """RMSE and NLPD from an ensemble of test predictions (Eqs. 66-68)."""
    mu = jnp.mean(test_samples, axis=0)
    var = jnp.var(test_samples, axis=0, ddof=1) + noise_var
    rmse = jnp.sqrt(jnp.mean((mu - y_test) ** 2))
    nlpd = jnp.mean(
        0.5 * jnp.log(2.0 * jnp.pi * var) + (y_test - mu) ** 2 / (2.0 * var)
    )
    return float(rmse), float(nlpd)
