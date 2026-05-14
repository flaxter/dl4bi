# Build the Stan source for a deepRV model. Supported v0.1 surface:
#   - exactly one deepRV() term
#   - fixed-effect design matrix X built from the formula RHS minus
#     the deepRV() term (intercept included by default)
#   - prior_uniform on ls; default N(0, 1) on b; default half-normal(0, 1)
#     on sigma (gaussian only)
#   - family in {poisson, gaussian}
#
# Each branch lives in one place: family-specific bits in build_stancode()
# and build_standata(); shared shape checks in standata. The decoder's
# decode function source is inlined verbatim from inst/stan/decode_mlp.stan
# so that file is the single source of truth.

read_decode_functions_block <- function() {
  path <- system.file("stan", "decode_mlp.stan", package = "brms.deeprv")
  if (!nzchar(path) || !file.exists(path)) {
    stop("inst/stan/decode_mlp.stan not found in the installed package",
         call. = FALSE)
  }
  src <- paste(readLines(path, warn = FALSE), collapse = "\n")
  m <- regmatches(src, regexpr("functions\\s*\\{[\\s\\S]*\\}", src, perl = TRUE))
  if (length(m) != 1L) {
    stop("could not locate `functions { ... }` block in decode_mlp.stan",
         call. = FALSE)
  }
  inner <- sub("^functions\\s*\\{", "", m)
  inner <- sub("\\}\\s*$", "", inner)
  trimws(inner)
}

# Family-specific Stan snippets. The `by_mode` controls the form of the
# spatial term inside the likelihood:
#   "none"   -> mu[obs_idx]
#   "svc"    -> x_by .* mu[obs_idx]              (element-wise SVC)
#   "factor" -> spatial[]   (Stan gathers per-obs from mu[group_idx, obs_idx])
family_spec <- function(family_name, by_mode = "none") {
  spatial_term <- switch(by_mode,
    none   = "mu[obs_idx]",
    svc    = "x_by .* mu[obs_idx]",
    factor = "spatial",
    stop(sprintf("unknown by_mode: %s", by_mode), call. = FALSE)
  )
  switch(family_name,
    poisson = list(
      y_decl    = "  array[N] int<lower=0> y;",
      extra_par = NULL,
      extra_prior = NULL,
      likelihood = sprintf("  target += poisson_log_lpmf(y | X * b + %s);",
                           spatial_term)
    ),
    gaussian = list(
      y_decl    = "  vector[N] y;",
      extra_par = "  real<lower=0> sigma;",
      extra_prior = "  sigma ~ std_normal();",
      likelihood = sprintf("  target += normal_lpdf(y | X * b + %s, sigma);",
                           spatial_term)
    ),
    stop(sprintf("unsupported family: %s", family_name), call. = FALSE)
  )
}

# Build the full Stan program. Dispatches to the Kronecker codepath
# when given a deepRV_decoder_kron; otherwise falls through to the 1D
# build below.
build_stancode <- function(decoder, ls_prior, family, by_mode = "none") {
  if (inherits(decoder, "deepRV_decoder_kron")) {
    if (by_mode != "none") {
      stop("by != NULL is not supported with Kronecker decoders in v0.1",
           call. = FALSE)
    }
    return(build_stancode_kron(decoder, ls_prior, family))
  }
  build_stancode_1d(decoder, ls_prior, family, by_mode)
}

# Build the 1D Stan program. ls_prior may be NULL for the (future)
# case where ls has no user prior; for now prior_uniform is required.
#
# by_mode controls how the decoder's mu contributes to eta:
#   "none"   : eta_n = X[n,:] %*% b + mu[obs_idx[n]]                  (default)
#   "svc"    : eta_n = X[n,:] %*% b + x_by[n] * mu[obs_idx[n]]        (SVC)
#   "factor" : matrix[G, L] z; matrix[G, L] mu; eta_n adds
#              mu[group_idx[n], obs_idx[n]]. Single shared ls.
build_stancode_1d <- function(decoder, ls_prior, family, by_mode = "none") {
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")
  stopifnot(by_mode %in% c("none", "svc", "factor"))
  fs <- family_spec(family, by_mode)
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)

  extra_data <- switch(by_mode,
    none   = NULL,
    svc    = "  vector[N] x_by;",
    factor = c("  int<lower=1> G;",
               "  array[N] int<lower=1, upper=G> group_idx;")
  )
  # Parameter declarations and transformed parameter body differ by mode:
  # for "factor" the latent z is matrix[G, L] and mu is matrix[G, L].
  if (by_mode == "factor") {
    z_par <- "  matrix[G, L] z;"
    z_prior <- "  to_vector(z) ~ std_normal();"
    mu_block <- c(
      "  vector[cond_dim] cond;",
      "  cond[1] = ls;",
      "  matrix[G, L] mu;",
      "  for (g in 1:G)",
      "    mu[g] = decode(to_vector(z[g]), cond, W1, b1, W2, b2)';"
    )
    # Stan needs the spatial vector built before it's used in the likelihood.
    pre_likelihood <- c(
      "  vector[N] spatial;",
      "  for (n in 1:N) spatial[n] = mu[group_idx[n], obs_idx[n]];"
    )
  } else {
    z_par <- "  vector[L] z;"
    z_prior <- "  z ~ std_normal();"
    mu_block <- c(
      "  vector[cond_dim] cond;",
      "  cond[1] = ls;",
      "  vector[L] mu = decode(z, cond, W1, b1, W2, b2);"
    )
    pre_likelihood <- NULL
  }

  lines <- c(
    "functions {",
    decode_body,
    "}",
    "data {",
    "  int<lower=1> N;",
    "  int<lower=1> L;",
    "  int<lower=1> cond_dim;",
    "  int<lower=1> hidden;",
    "  int<lower=1> K;",
    "  matrix[N, K] X;",
    "  matrix[L + cond_dim, hidden] W1;",
    "  vector[hidden]               b1;",
    "  matrix[hidden, L]            W2;",
    "  vector[L]                    b2;",
    fs$y_decl,
    "  array[N] int<lower=1, upper=L>  obs_idx;",
    extra_data,
    "}",
    "parameters {",
    z_par,
    sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi),
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    mu_block,
    "}",
    "model {",
    z_prior,
    "  b ~ std_normal();",
    fs$extra_prior,
    sprintf("  // ls ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_lo, ls_hi),
    pre_likelihood,
    fs$likelihood,
    "}"
  )
  paste(Filter(Negate(is.null), lines), collapse = "\n")
}

# Kronecker Stan program. Latent Z is matrix[N_side, N_side]; apply the
# axis decoder to each column with conditional ls_x, then to each row of
# the result with ls_y. Flatten column-major into length-L mu so the
# obs_idx -> grid mapping matches load_deeprv_kron()'s grid_coords
# convention (col 1 varies fastest = x, then y).
build_stancode_kron <- function(decoder, ls_prior, family) {
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")
  stopifnot(inherits(decoder, "deepRV_decoder_kron"))
  fs <- family_spec(family, by_mode = "none")
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)

  lines <- c(
    "functions {",
    decode_body,
    "}",
    "data {",
    "  int<lower=1> N;",
    "  int<lower=1> L;",          # = N_side * N_side
    "  int<lower=1> N_side;",
    "  int<lower=1> cond_dim;",
    "  int<lower=1> hidden;",
    "  int<lower=1> K;",
    "  matrix[N, K] X;",
    "  matrix[N_side + cond_dim, hidden] W1;",   # axis decoder
    "  vector[hidden]                    b1;",
    "  matrix[hidden, N_side]            W2;",
    "  vector[N_side]                    b2;",
    fs$y_decl,
    "  array[N] int<lower=1, upper=L>  obs_idx;",
    "}",
    "parameters {",
    "  matrix[N_side, N_side] z;",
    sprintf("  real<lower=%s, upper=%s> ls_x;", ls_lo, ls_hi),
    sprintf("  real<lower=%s, upper=%s> ls_y;", ls_lo, ls_hi),
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    "  vector[cond_dim] cond_x;",
    "  vector[cond_dim] cond_y;",
    "  cond_x[1] = ls_x;",
    "  cond_y[1] = ls_y;",
    # Apply axis decoder along columns of z -> mid (N_side x N_side).
    "  matrix[N_side, N_side] mid;",
    "  for (j in 1:N_side)",
    "    mid[, j] = decode(z[, j], cond_x, W1, b1, W2, b2);",
    # Then along rows of mid -> mu_grid (N_side x N_side).
    "  matrix[N_side, N_side] mu_grid;",
    "  for (i in 1:N_side)",
    "    mu_grid[i] = decode(to_vector(mid[i]), cond_y, W1, b1, W2, b2)';",
    # Flatten column-major; flat index (j-1) * N_side + i picks mu_grid[i, j].
    "  vector[L] mu = to_vector(mu_grid);",
    "}",
    "model {",
    "  to_vector(z) ~ std_normal();",
    "  b ~ std_normal();",
    fs$extra_prior,
    sprintf("  // ls_x, ls_y ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_lo, ls_hi),
    fs$likelihood,
    "}"
  )
  paste(Filter(Negate(is.null), lines), collapse = "\n")
}

# Build the standata list. Handles:
#   - obs_idx range and length checks
#   - family-specific y type coercion
#   - design matrix X (defaults to intercept-only if rhs_formula is NULL)
#   - by_values for SVC (passed in as `x_by`) or factor (passed as
#     integer group_idx + G)
build_standata <- function(decoder, deepRV_call, y, X, family,
                           by_mode = "none", by_values = NULL,
                           n_groups = NULL) {
  is_kron <- inherits(decoder, "deepRV_decoder_kron")
  obs_idx <- as.integer(deepRV_call$obs_idx)
  if (length(obs_idx) != length(y)) {
    stop(sprintf(
      "obs_idx (length %d) must have the same length as the response y (length %d)",
      length(obs_idx), length(y)),
      call. = FALSE)
  }
  if (any(is.na(obs_idx)) || any(obs_idx < 1L) || any(obs_idx > decoder$L)) {
    stop(sprintf(
      "obs_idx values must be integers in [1, L=%d]; got range [%s, %s]%s",
      decoder$L,
      if (length(obs_idx) > 0) min(obs_idx, na.rm = TRUE) else "?",
      if (length(obs_idx) > 0) max(obs_idx, na.rm = TRUE) else "?",
      if (any(is.na(obs_idx))) " (with NAs)" else ""),
      call. = FALSE)
  }
  if (!is.matrix(X) || nrow(X) != length(y)) {
    stop(sprintf("design matrix X must have nrow == length(y); got %s x %d vs %d",
                 paste(dim(X), collapse = "x"), ncol(X), length(y)),
         call. = FALSE)
  }
  if (ncol(X) < 1L) {
    stop("design matrix X must have at least one column", call. = FALSE)
  }
  if (any(is.na(X))) {
    stop("design matrix X contains NAs; drop or impute incomplete rows first",
         call. = FALSE)
  }

  y_data <- switch(family,
    poisson = {
      if (!is.numeric(y) || any(y < 0) || any(y != round(y))) {
        stop("Poisson family requires non-negative integer y", call. = FALSE)
      }
      as.integer(y)
    },
    gaussian = {
      if (!is.numeric(y)) stop("Gaussian family requires numeric y", call. = FALSE)
      if (any(is.na(y))) stop("y contains NAs", call. = FALSE)
      as.numeric(y)
    },
    stop(sprintf("unsupported family: %s", family), call. = FALSE)
  )

  if (is_kron) {
    # Kronecker: axis decoder weights (shared along x and y) come from the
    # x_decoder slot. cond_dim refers to one axis (length 1 for v0.1).
    axis_dec <- decoder$x_decoder
    out <- list(
      N        = length(y),
      L        = as.integer(decoder$L),
      N_side   = as.integer(decoder$grid_side),
      cond_dim = length(axis_dec$conditionals),
      hidden   = ncol(axis_dec$weights$W1),
      K        = ncol(X),
      X        = X,
      W1       = axis_dec$weights$W1,
      b1       = as.array(axis_dec$weights$b1),
      W2       = axis_dec$weights$W2,
      b2       = as.array(axis_dec$weights$b2),
      y        = y_data,
      obs_idx  = obs_idx
    )
  } else {
    out <- list(
      N        = length(y),
      L        = as.integer(decoder$L),
      cond_dim = length(decoder$conditionals),
      hidden   = ncol(decoder$weights$W1),
      K        = ncol(X),
      X        = X,
      W1       = decoder$weights$W1,
      b1       = as.array(decoder$weights$b1),
      W2       = decoder$weights$W2,
      b2       = as.array(decoder$weights$b2),
      y        = y_data,
      obs_idx  = obs_idx
    )
  }
  if (by_mode == "svc") {
    if (is.null(by_values) || length(by_values) != length(y)) {
      stop("by_values must have the same length as y for SVC mode",
           call. = FALSE)
    }
    out$x_by <- as.numeric(by_values)
  } else if (by_mode == "factor") {
    if (is.null(by_values) || length(by_values) != length(y)) {
      stop("by_values (group_idx) must have the same length as y for ",
           "factor mode", call. = FALSE)
    }
    if (is.null(n_groups) || n_groups < 1L) {
      stop("n_groups must be a positive integer for factor mode",
           call. = FALSE)
    }
    out$G <- as.integer(n_groups)
    out$group_idx <- as.integer(by_values)
  }
  out
}
