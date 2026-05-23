"""FlowGP: sampling Gaussian process predictives under arbitrary conditioning.

Replication of the whitened probability-flow ODE sampler (Algorithm 1) from
"Conditioning Gaussian Processes on Almost Anything" (Moss, Astfalck et al.,
arXiv:2605.21041). The authors did not release code; this is an independent
reimplementation in JAX.

The method recasts GP predictive sampling as a variance-preserving diffusion.
Given a linear-Gaussian predictive N(m_pred, K_pred) and a point-wise evaluable
(differentiable) log-likelihood log p(C | f0) encoding non-linear / non-Gaussian
conditions C, samples from p(f0 | D, C) are obtained by integrating a guided
probability-flow ODE backwards in time. Whitening removes the Gaussian dynamics
so only the guidance correction remains (Section 5.1, Eq. 19).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def make_schedule(beta0: float = 1e-5, beta1: float = 10.0):
    """Linear VP beta-schedule and corresponding signal scale alpha (Appendix F.1).

    beta(t) = beta0 + (beta1 - beta0) t
    alpha(t) = exp(-0.5 * int_0^t beta(s) ds) = exp(-0.5 beta0 t - 0.25 (beta1-beta0) t^2)
    """

    def beta(t):
        return beta0 + (beta1 - beta0) * t

    def alpha(t):
        return jnp.exp(-(0.5 * beta0 * t + 0.25 * (beta1 - beta0) * t**2))

    def log_snr(t):
        a = alpha(t)
        return jnp.log(a) - 0.5 * jnp.log1p(-(a**2) + 1e-8)

    return beta, alpha, log_snr


def snr_uniform_grid(
    log_snr: Callable,
    T: int,
    t_max: float = 1.0,
    t_min: float = 1e-3,
    n_bisect: int = 60,
) -> jnp.ndarray:
    """Decreasing time grid 1 = t_0 > ... > t_T ~ 0, uniform in log-SNR (Appendix F.2).

    SNR is monotonically decreasing in t, so we place T+1 grid points at equal
    intervals in log-SNR and invert via bisection. This concentrates steps near
    t = 0 where the dynamics are stiffest (Theorem D.1).
    """
    targets = jnp.linspace(log_snr(t_max), log_snr(t_min), T + 1)

    def invert(target):
        def body(_, c):
            lo, hi = c
            mid = 0.5 * (lo + hi)
            # log_snr decreasing in t: if current SNR exceeds target, move to larger t.
            move_up = log_snr(mid) > target
            lo = jnp.where(move_up, mid, lo)
            hi = jnp.where(move_up, hi, mid)
            return (lo, hi)

        lo, hi = jax.lax.fori_loop(0, n_bisect, body, (t_min, t_max))
        return 0.5 * (lo + hi)

    ts = jax.vmap(invert)(targets)
    # Pin the endpoints exactly.
    ts = ts.at[0].set(t_max).at[-1].set(t_min)
    return ts


def flowgp_sample(
    key: jax.Array,
    m_pred: jnp.ndarray,
    K_pred: jnp.ndarray,
    loglik_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_samples: int = 100,
    T: int = 1000,
    S: int = 5,
    v_max: float = 1e2,
    jitter: float = 1e-6,
    beta0: float = 1e-5,
    beta1: float = 10.0,
    t_min: float = 1e-3,
) -> jnp.ndarray:
    """Draw samples from p(f0 | D, C) via the whitened probability-flow ODE.

    Args:
        key: PRNG key.
        m_pred: (m,) linear-Gaussian predictive mean m_{*|y}.
        K_pred: (m, m) linear-Gaussian predictive covariance K_{**|y}.
        loglik_fn: maps a single f0 of shape (m,) to a scalar log p(C | f0).
            Must be differentiable under JAX autodiff.
        n_samples: number of independent predictive samples (ODE trajectories).
        T: number of Euler steps.
        S: Monte Carlo samples for the guidance term per step.
        v_max: smooth-clipping threshold on the whitened velocity (Algorithm 1, v).
        jitter: diagonal jitter added to K_pred before the Cholesky factorisation.
        beta0, beta1: VP schedule endpoints.
        t_min: truncation tau, avoiding the singular bridge factor as t -> 0.

    Returns:
        (n_samples, m) array of samples from the conditioned predictive.
    """
    beta, alpha, log_snr = make_schedule(beta0, beta1)
    m = m_pred.shape[0]

    # Whitening: K_pred = L L^T, with unwhitening f0 = L f_hat + m_pred (Algorithm 1, line 1).
    L = jnp.linalg.cholesky(K_pred + jitter * jnp.eye(m))

    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts = ts[:-1] - ts[1:]  # (T,) positive step sizes
    alphas = alpha(ts[:-1])  # (T,)
    betas = beta(ts[:-1])  # (T,)

    key_init, key_eps = jax.random.split(key)
    # Initial whitened state: pure white noise (Algorithm 1, line 3).
    f_hat = jax.random.normal(key_init, (n_samples, m))
    # Reparameterisation trick: fix the MC noise across steps to reduce
    # step-to-step variance (Appendix F.3).
    eps = jax.random.normal(key_eps, (n_samples, S, m))

    def whitened_loglik(fh):  # fh: (m,) whitened sample of f0
        return loglik_fn(L @ fh + m_pred)

    # Returns log p(C | f0) and its gradient w.r.t. the *whitened* f0 (= L^T grad_f0).
    val_and_grad = jax.vmap(jax.value_and_grad(whitened_loglik))

    def step(f_hat, inputs):
        dt, a, b = inputs
        # (i) S samples from f0_hat | f_hat, D ~ N(a f_hat, (1 - a^2) I) (Eq. 25).
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps  # (N, S, m)

        ll, s = val_and_grad(f0_hat.reshape(-1, m))
        ll = ll.reshape(n_samples, S)  # log-likelihoods
        s = s.reshape(n_samples, S, m)  # whitened-space scores

        # (ii) self-normalised importance weights, numerically stable (logsumexp).
        w = jax.nn.softmax(ll, axis=1)  # (N, S)

        # (iv) guided velocity in whitened space (Eq. 19 with MC guidance Eq. 18/30).
        guidance = jnp.einsum("ns,nsm->nm", w, s)
        v = -0.5 * b * a * guidance  # (N, m)

        # (v) smooth clipping of the step for numerical stability (Algorithm 1, line 14).
        norm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(norm / v_max) / (norm + 1e-8) * v

        # (vi) explicit Euler step, integrating from t = 1 to t = 0.
        f_hat = f_hat - dt * v
        return f_hat, None

    f_hat, _ = jax.lax.scan(step, f_hat, (dts, alphas, betas))

    # Unwhiten the final state (Algorithm 1, line 17).
    return f_hat @ L.T + m_pred
