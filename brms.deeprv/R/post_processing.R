# Posterior predictive methods for deeprv_fit. Implemented as S3
# methods on the rstantools generics so `posterior_predict(fit)` and
# `posterior_epred(fit)` dispatch to our code when rstantools is on
# the search path (which it is via brms / rstanarm / our own Imports).
#
# v0.1 limitation (DESIGN.md section 2.7.1): prediction only at the
# training grid points carried in the fit's `obs_idx`. The optional
# `newdata` argument is reserved for v0.2 - calling with newdata errors
# loudly with a pointer to brms::gp().

#' Compute draws of the linear predictor eta = X * b + mu[obs_idx]
#'
#' Useful as a primitive for custom post-processing; both
#' [posterior_epred()] and [posterior_predict()] go through this.
#'
#' @param fit A `deeprv_fit` from [deeprv_brm()].
#' @return Matrix of (S draws) x (N observations) on the link scale.
#' @export
posterior_eta_draws <- function(fit) {
  if (!inherits(fit, "deeprv_fit")) {
    stop("posterior_eta_draws() requires a deeprv_fit object", call. = FALSE)
  }
  X <- fit$X                # (N, K)
  obs_idx <- as.integer(fit$standata$obs_idx)
  by_mode <- if (is.null(fit$by_mode)) "none" else fit$by_mode

  if (inherits(fit$decoder, "deepRV_decoder_kron")) {
    draws <- rstan::extract(fit$stanfit, pars = c("z", "ls_x", "ls_y", "b"))
    b <- as.matrix(draws$b)
    fixed <- b %*% t(X)
    mu_full <- forward_decode_kron_batched(fit$decoder, draws$z,
                                           draws$ls_x, draws$ls_y)
    spatial <- mu_full[, obs_idx, drop = FALSE]
    return(fixed + spatial)
  }

  if (identical(by_mode, "st")) {
    dec_t <- fit$deepRV$decoder_time
    is_kron <- inherits(fit$decoder, "deepRV_decoder_kron")
    pars <- c("z", "b")
    if (is_kron) {
      pars <- c(pars, "ls_x", "ls_y")
    } else {
      pars <- c(pars, "ls")
    }
    has_phi     <- "phi"     %in% fit$stanfit@model_pars
    has_sigma_t <- "sigma_t" %in% fit$stanfit@model_pars
    has_ls_t    <- "ls_t"    %in% fit$stanfit@model_pars
    if (has_phi)     pars <- c(pars, "phi")
    if (has_sigma_t) pars <- c(pars, "sigma_t")
    if (has_ls_t)    pars <- c(pars, "ls_t")
    draws <- rstan::extract(fit$stanfit, pars = pars)
    b <- as.matrix(draws$b)
    fixed <- b %*% t(X)
    time_idx <- as.integer(fit$standata$time_idx)
    F_draws <- forward_st_batched(
      fit$decoder, draws$z,
      ls_draws    = if (is_kron) NULL else draws$ls,
      ls_x_draws  = if (is_kron) draws$ls_x else NULL,
      ls_y_draws  = if (is_kron) draws$ls_y else NULL,
      sigma_t_draws = if (has_sigma_t) draws$sigma_t else NULL,
      phi_draws     = if (has_phi)     draws$phi     else NULL,
      decoder_time  = if (inherits(dec_t, "deepRV_decoder")) dec_t else NULL,
      ls_t_draws    = if (has_ls_t)    draws$ls_t    else NULL
    )
    S <- dim(F_draws)[1L]
    N <- length(time_idx)
    spatial <- matrix(0, nrow = S, ncol = N)
    for (n in seq_len(N)) {
      spatial[, n] <- F_draws[, time_idx[n], obs_idx[n]]
    }
    return(fixed + spatial)
  }

  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls", "b"))
  ls <- draws$ls            # (S,)
  b <- as.matrix(draws$b)   # (S, K)
  fixed <- b %*% t(X)       # (S, N)

  if (by_mode == "factor") {
    # z is matrix[G, L] per Stan: extract pulls (S, G, L). When
    # ls_pooling != "complete" ls is vector[G] in Stan so draws$ls is
    # (S, G); for "complete" it's (S,) and shared across groups.
    z <- draws$z                    # (S, G, L)
    S <- dim(z)[1L]; G <- dim(z)[2L]; L <- dim(z)[3L]
    group_idx <- as.integer(fit$standata$group_idx)
    N <- length(group_idx)
    spatial <- matrix(0, nrow = S, ncol = N)
    ls_is_per_group <- length(dim(ls)) == 2L
    for (g in seq_len(G)) {
      ls_g <- if (ls_is_per_group) ls[, g] else ls
      mu_g <- forward_decode_batched(fit$decoder, z[, g, ], ls_g)
      cols <- which(group_idx == g)
      if (length(cols) > 0L) {
        spatial[, cols] <- mu_g[, obs_idx[cols], drop = FALSE]
      }
    }
    return(fixed + spatial)
  }

  z <- draws$z                                          # (S, L)
  mu_full <- forward_decode_batched(fit$decoder, z, ls) # (S, L)
  spatial <- mu_full[, obs_idx, drop = FALSE]           # (S, N)
  if (by_mode == "svc") {
    spatial <- sweep(spatial, 2L, as.numeric(fit$by_values), FUN = "*")
  }
  fixed + spatial
}

# Batched forward pass through MLPDeepRV.decode for S draws.
# Pure R; avoids re-compiling Stan or paying the expose_stan_functions
# tax for every posterior call. Matches mlp_decode.stan's `decode`:
#   h  = relu(W1' * x + b1) where x = [z; cond]
#   mu = W2' * h + b2
forward_decode_batched <- function(decoder, z_draws, ls_draws) {
  # Allow Kron decoders by transparently using the axis decoder weights.
  if (inherits(decoder, "deepRV_decoder_kron")) decoder <- decoder$x_decoder
  W1 <- decoder$weights$W1
  b1 <- decoder$weights$b1
  W2 <- decoder$weights$W2
  b2 <- decoder$weights$b2
  if (!is.matrix(z_draws)) z_draws <- matrix(z_draws, nrow = 1L)
  x_aug <- cbind(z_draws, as.numeric(ls_draws))  # (S, L + 1)
  pre_h <- sweep(x_aug %*% W1, 2L, b1, FUN = "+")
  h <- pmax(pre_h, 0)
  sweep(h %*% W2, 2L, b2, FUN = "+")             # (S, L)
}

# Space-time forward for the RW time prior. Given S draws of
# z (S, T, L), ls (S,), and sigma_t (S,), returns F (S, T, L) where
# eps[s, t] = decode(z[s, t], ls[s]) and
# F[s, t] = F[s, t-1] + sigma_t[s] * eps[s, t] with F[s, 0] = 0.
forward_st_batched <- function(decoder, z_draws, ls_draws = NULL,
                               sigma_t_draws = NULL,
                               phi_draws = NULL,
                               decoder_time = NULL, ls_t_draws = NULL,
                               ls_x_draws = NULL, ls_y_draws = NULL) {
  S <- dim(z_draws)[1L]
  T_full <- dim(z_draws)[2L]
  L <- dim(z_draws)[3L]
  # Step 1: spatial decode per time step -> eps (S, T, L).
  eps <- array(0, dim = c(S, T_full, L))
  is_kron <- inherits(decoder, "deepRV_decoder_kron")
  if (is_kron) {
    N_side <- as.integer(decoder$grid_side)
    for (t in seq_len(T_full)) {
      # Reshape (S, L) -> (S, N_side, N_side) column-major: matches Stan's
      # to_matrix(z[t], N_side, N_side) ordering.
      zt <- array(z_draws[, t, ], dim = c(S, N_side, N_side))
      eps[, t, ] <- forward_decode_kron_batched(decoder, zt,
                                                 ls_x_draws, ls_y_draws)
    }
  } else {
    for (t in seq_len(T_full)) {
      eps[, t, ] <- forward_decode_batched(decoder, z_draws[, t, ], ls_draws)
    }
  }
  # Step 2: combine in time.
  if (inherits(decoder_time, "deepRV_decoder")) {
    # Apply the time decoder column-wise: for each spatial point j and
    # each draw s, F[s, :, j] = decode_t(eps[s, :, j], ls_t[s]).
    F_draws <- array(0, dim = c(S, T_full, L))
    for (j in seq_len(L)) {
      # eps[, , j] is (S, T) — exactly the shape forward_decode_batched wants.
      F_draws[, , j] <- forward_decode_batched(decoder_time, eps[, , j],
                                                ls_t_draws)
    }
    return(F_draws)
  }
  # rw / ar1 path: scalar sigma_t, optional phi (ar1).
  F_draws <- array(0, dim = c(S, T_full, L))
  ar1 <- !is.null(phi_draws)
  for (t in seq_len(T_full)) {
    eps_t <- eps[, t, ]
    scaled <- sweep(eps_t, 1L, sigma_t_draws, FUN = "*")
    if (t == 1L) {
      if (ar1) {
        init_scale <- 1 / sqrt(1 - phi_draws^2)
        F_t <- sweep(scaled, 1L, init_scale, FUN = "*")
      } else {
        F_t <- scaled
      }
    } else {
      prev <- F_draws[, t - 1L, ]
      if (ar1) prev <- sweep(prev, 1L, phi_draws, FUN = "*")
      F_t <- prev + scaled
    }
    F_draws[, t, ] <- F_t
  }
  F_draws
}

# Kronecker forward for S draws of z (S, N_side, N_side), ls_x (S,),
# ls_y (S,). Returns (S, L) where L = N_side^2 and the flattening is
# column-major to match the load_deeprv_kron() grid_coords order.
#
# Per draw s, mu_grid[i, j] = decode(decode(z[, j], ls_x)[i], ls_y).
# Equivalently: apply forward_decode_batched along columns of z (vectorised
# over draws), then along rows of the result.
forward_decode_kron_batched <- function(decoder, z_draws, ls_x, ls_y) {
  axis <- decoder$x_decoder
  N <- as.integer(decoder$grid_side)
  S <- dim(z_draws)[1L]
  # mid[s, i, j] = decode(z[s, , j], ls_x[s])[i]
  # Apply per-column: for each fixed j, z[, , j] is (S, N).
  mid <- array(0, dim = c(S, N, N))
  for (j in seq_len(N)) {
    mid[, , j] <- forward_decode_batched(axis, z_draws[, , j], ls_x)
  }
  # mu_grid[s, i, j] = decode(mid[s, i, ], ls_y[s])[j]
  # Apply per-row: for each fixed i, mid[, i, ] is (S, N).
  mu_grid <- array(0, dim = c(S, N, N))
  for (i in seq_len(N)) {
    mu_grid[, i, ] <- forward_decode_batched(axis, mid[, i, ], ls_y)
  }
  # R stores arrays with first index fastest, so the (s, i, j) storage
  # already lays elements out as (s=1..S, i=1..N, j=1..N) - and that's
  # exactly column-major (j slowest, i then s) for the flat 2D grid
  # flat_idx = (j - 1) * N + i. matrix(mu_grid, S, N*N) reads off rows of
  # S draws, columns in flat_idx order.
  matrix(mu_grid, nrow = S, ncol = N * N)
}

# Helper: enforce v0.1's "no new locations" rule. Once newdata support
# lands (v0.2), this guard splits into "newdata is on the grid" vs
# "use snap_to_grid()".
check_no_newdata <- function(newdata) {
  if (!is.null(newdata)) {
    stop("deepRV decoders only support prediction at the training grid ",
         "points. Pass newdata = NULL to predict at the rows used in the ",
         "fit, or use brms::gp() if you need new-location prediction.",
         call. = FALSE)
  }
}

#' Posterior expected response for a deeprv_fit
#'
#' @param object A `deeprv_fit` from [deeprv_brm()].
#' @param newdata Must be NULL in v0.1 - new-location prediction is
#'   not supported with `MLPDeepRV` (see `DESIGN.md` section 2.7.1).
#' @param ... Unused.
#' @return Matrix (S x N) of posterior draws of the expected response
#'   on the response scale (e.g. `exp(eta)` for Poisson).
#' @export
posterior_epred.deeprv_fit <- function(object, newdata = NULL, ...) {
  check_no_newdata(newdata)
  eta <- posterior_eta_draws(object)
  switch(object$family$family,
    poisson  = exp(eta),
    gaussian = eta,
    stop("unsupported family in posterior_epred: ", object$family$family,
         call. = FALSE)
  )
}

#' Posterior predictive draws for a deeprv_fit
#'
#' @param object A `deeprv_fit` from [deeprv_brm()].
#' @param newdata Must be NULL in v0.1.
#' @param ... Unused.
#' @return Matrix (S x N) of posterior predictive draws on the
#'   response scale, sampled from the family.
#' @export
posterior_predict.deeprv_fit <- function(object, newdata = NULL, ...) {
  check_no_newdata(newdata)
  eta <- posterior_eta_draws(object)
  fam <- object$family$family
  if (fam == "poisson") {
    return(matrix(stats::rpois(length(eta), exp(eta)),
                  nrow = nrow(eta), ncol = ncol(eta)))
  }
  if (fam == "gaussian") {
    sigma <- rstan::extract(object$stanfit, pars = "sigma")$sigma  # (S,)
    noise <- matrix(stats::rnorm(length(eta)),
                    nrow = nrow(eta), ncol = ncol(eta))
    return(eta + sigma * noise)
  }
  stop("unsupported family in posterior_predict: ", fam, call. = FALSE)
}
