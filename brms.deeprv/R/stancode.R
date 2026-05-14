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
# Build the Stan source for a space-time deepRV term.
#
# decoder_time may be:
#   "rw"   — random walk in time: F[t] = F[t-1] + sigma_t * eps[t]
#   "ar1"  — lag-1 AR with stationary marginal
#            F[1] = (sigma_t/sqrt(1-phi^2)) * eps[1]; F[t] = phi*F[t-1] + sigma_t*eps[t]
#   <deepRV_decoder> — apply the 1D time decoder column-wise to eps to
#                      produce F. Routes to build_stancode_st_decoder_time().
build_stancode_st <- function(decoder, ls_prior, sigma_t_prior, family,
                              decoder_time = "rw", ls_t_prior = NULL) {
  if (inherits(decoder, "deepRV_decoder_kron")) {
    return(build_stancode_st_kron(decoder, ls_prior, sigma_t_prior, family,
                                  decoder_time, ls_t_prior))
  }
  if (inherits(decoder_time, "deepRV_decoder")) {
    if (is.null(ls_t_prior)) {
      stop("when decoder_time is a deepRV_decoder, ls_t_prior is required",
           call. = FALSE)
    }
    return(build_stancode_st_decoder_time(decoder, ls_prior, sigma_t_prior,
                                          decoder_time, ls_t_prior, family))
  }
  stopifnot(decoder_time %in% c("rw", "ar1"))
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(inherits(sigma_t_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")
  fs <- family_spec(family, by_mode = "none")
  # Replace the spatial term in the likelihood with our gathered vector.
  fs$likelihood <- sub("mu\\[obs_idx\\]", "spatial", fs$likelihood)
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)
  sigma_t_prior_line <- stan_prior_statement(sigma_t_prior, "sigma_t")

  # Time prior fragments: parameter declarations, F recurrence, prior.
  if (decoder_time == "rw") {
    time_pars <- NULL          # only sigma_t (declared below)
    time_priors <- NULL
    F_block <- c(
      "  matrix[T, L] F;",
      "  F[1] = sigma_t * eps[1];",
      "  for (t in 2:T) F[t] = F[t - 1] + sigma_t * eps[t];"
    )
  } else {                     # ar1
    time_pars <- "  real<lower=-1, upper=1> phi;"
    # Uniform on (-1, 1) is implicit from bounds; no extra statement.
    time_priors <- "  // phi ~ uniform(-1, 1) -- implicit from parameter bounds"
    F_block <- c(
      "  matrix[T, L] F;",
      # Stationary initialisation: marginal Var(F[1]) = sigma_t^2 / (1 - phi^2)
      "  F[1] = (sigma_t / sqrt(1 - square(phi))) * eps[1];",
      "  for (t in 2:T) F[t] = phi * F[t - 1] + sigma_t * eps[t];"
    )
  }

  lines <- c(
    "functions {",
    decode_body,
    "}",
    "data {",
    "  int<lower=1> N;",
    "  int<lower=1> L;",
    "  int<lower=1> T;",
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
    "  array[N] int<lower=1, upper=T>  time_idx;",
    "}",
    "parameters {",
    "  matrix[T, L] z;",
    sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi),
    "  real<lower=0> sigma_t;",
    time_pars,
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    "  vector[cond_dim] cond;",
    "  cond[1] = ls;",
    "  matrix[T, L] eps;",
    "  for (t in 1:T) eps[t] = decode(to_vector(z[t]), cond, W1, b1, W2, b2)';",
    F_block,
    "}",
    "model {",
    "  to_vector(z) ~ std_normal();",
    "  b ~ std_normal();",
    fs$extra_prior,
    sigma_t_prior_line,
    time_priors,
    sprintf("  // ls ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_lo, ls_hi),
    "  vector[N] spatial;",
    "  for (n in 1:N) spatial[n] = F[time_idx[n], obs_idx[n]];",
    fs$likelihood,
    "}"
  )
  paste(Filter(Negate(is.null), lines), collapse = "\n")
}

# Space-time program with a decoder-based time prior. Both space and time
# decoders are 1D MLPs; the spatial one operates per time step, the time
# one operates per spatial point. Equivalent to sampling from a separable
# space-time GP F = decode_t(decode_s(Z, ls_s), ls_t).
build_stancode_st_decoder_time <- function(decoder, ls_prior, sigma_t_prior,
                                           decoder_time, ls_t_prior, family) {
  stopifnot(inherits(decoder, "deepRV_decoder"))
  stopifnot(!inherits(decoder, "deepRV_decoder_kron"))
  stopifnot(inherits(decoder_time, "deepRV_decoder"))
  stopifnot(!inherits(decoder_time, "deepRV_decoder_kron"))
  stopifnot(inherits(ls_prior, "deepRV_prior") && ls_prior$family == "uniform")
  stopifnot(inherits(ls_t_prior, "deepRV_prior") && ls_t_prior$family == "uniform")
  validate_prior_in_range(ls_t_prior, decoder_time$ls_trained_range,
                          decoder_label = sprintf("<time decoder %s>",
                                                  decoder_time$kernel))
  fs <- family_spec(family, by_mode = "none")
  fs$likelihood <- sub("mu\\[obs_idx\\]", "spatial", fs$likelihood)
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)
  ls_t_lo <- sprintf("%.10g", ls_t_prior$lower)
  ls_t_hi <- sprintf("%.10g", ls_t_prior$upper)

  lines <- c(
    "functions {",
    decode_body,
    "}",
    "data {",
    "  int<lower=1> N;",
    "  int<lower=1> L;",
    "  int<lower=1> T;",
    "  int<lower=1> cond_dim;",
    "  int<lower=1> hidden;",
    "  int<lower=1> cond_dim_t;",
    "  int<lower=1> hidden_t;",
    "  int<lower=1> K;",
    "  matrix[N, K] X;",
    # Space decoder weights.
    "  matrix[L + cond_dim, hidden] W1;",
    "  vector[hidden]               b1;",
    "  matrix[hidden, L]            W2;",
    "  vector[L]                    b2;",
    # Time decoder weights.
    "  matrix[T + cond_dim_t, hidden_t] W1_t;",
    "  vector[hidden_t]                 b1_t;",
    "  matrix[hidden_t, T]              W2_t;",
    "  vector[T]                        b2_t;",
    fs$y_decl,
    "  array[N] int<lower=1, upper=L>  obs_idx;",
    "  array[N] int<lower=1, upper=T>  time_idx;",
    "}",
    "parameters {",
    "  matrix[T, L] z;",
    sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi),
    sprintf("  real<lower=%s, upper=%s> ls_t;", ls_t_lo, ls_t_hi),
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    # Step 1: spatial decode per time step.
    "  vector[cond_dim] cond;",
    "  cond[1] = ls;",
    "  matrix[T, L] eps;",
    "  for (t in 1:T) eps[t] = decode(to_vector(z[t]), cond, W1, b1, W2, b2)';",
    # Step 2: time decode per spatial point.
    "  vector[cond_dim_t] cond_t;",
    "  cond_t[1] = ls_t;",
    "  matrix[T, L] F;",
    "  for (j in 1:L) F[, j] = decode(eps[, j], cond_t, W1_t, b1_t, W2_t, b2_t);",
    "}",
    "model {",
    "  to_vector(z) ~ std_normal();",
    "  b ~ std_normal();",
    fs$extra_prior,
    sprintf("  // ls   ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_lo, ls_hi),
    sprintf("  // ls_t ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_t_lo, ls_t_hi),
    "  vector[N] spatial;",
    "  for (n in 1:N) spatial[n] = F[time_idx[n], obs_idx[n]];",
    fs$likelihood,
    "}"
  )
  paste(Filter(Negate(is.null), lines), collapse = "\n")
}

# Helper: emit (parameters, model-block-priors, transformed-parameter-F)
# fragments for a given time prior. Used by build_stancode_st and
# build_stancode_st_kron so the time logic doesn't duplicate.
#
# Returns a list with:
#   pars        : extra parameter declarations beyond sigma_t (which is added
#                 by all rw/ar1 callers). NULL when decoder_time is a decoder.
#   priors      : model-block prior statements.
#   F_block_post_eps : Stan lines computing matrix[T, L] F given matrix[T, L] eps,
#                     sigma_t (for rw/ar1), and ls_t/decoder weights (for decoder).
time_block <- function(decoder_time, ls_t_prior) {
  if (inherits(decoder_time, "deepRV_decoder")) {
    ls_t_lo <- sprintf("%.10g", ls_t_prior$lower)
    ls_t_hi <- sprintf("%.10g", ls_t_prior$upper)
    list(
      pars = sprintf("  real<lower=%s, upper=%s> ls_t;", ls_t_lo, ls_t_hi),
      priors = sprintf(
        "  // ls_t ~ uniform(%s, %s)  -- implicit from parameter bounds",
        ls_t_lo, ls_t_hi),
      include_sigma_t = FALSE,
      F_block_post_eps = c(
        "  vector[cond_dim_t] cond_t;",
        "  cond_t[1] = ls_t;",
        "  matrix[T, L] F;",
        "  for (j in 1:L) F[, j] = decode(eps[, j], cond_t, W1_t, b1_t, W2_t, b2_t);"
      )
    )
  } else if (identical(decoder_time, "ar1")) {
    list(
      pars = "  real<lower=-1, upper=1> phi;",
      priors = "  // phi ~ uniform(-1, 1) -- implicit from parameter bounds",
      include_sigma_t = TRUE,
      F_block_post_eps = c(
        "  matrix[T, L] F;",
        "  F[1] = (sigma_t / sqrt(1 - square(phi))) * eps[1];",
        "  for (t in 2:T) F[t] = phi * F[t - 1] + sigma_t * eps[t];"
      )
    )
  } else {                       # "rw"
    list(
      pars = NULL,
      priors = NULL,
      include_sigma_t = TRUE,
      F_block_post_eps = c(
        "  matrix[T, L] F;",
        "  F[1] = sigma_t * eps[1];",
        "  for (t in 2:T) F[t] = F[t - 1] + sigma_t * eps[t];"
      )
    )
  }
}

# Kronecker spatial + any time prior. Spatial latent z is matrix[T, L]
# with L = N_side^2; per time step it's reshaped to N_side x N_side
# (column-major to match load_deeprv_kron's grid order), passed
# through the axis decoder along columns with ls_x then along rows
# with ls_y, then flattened back to length L as eps[t]. Time block
# (rw / ar1 / decoder) applies as in the 1D case.
build_stancode_st_kron <- function(decoder, ls_prior, sigma_t_prior, family,
                                   decoder_time = "rw", ls_t_prior = NULL) {
  stopifnot(inherits(decoder, "deepRV_decoder_kron"))
  stopifnot(inherits(ls_prior, "deepRV_prior") && ls_prior$family == "uniform")
  if (inherits(decoder_time, "deepRV_decoder") && is.null(ls_t_prior)) {
    stop("when decoder_time is a deepRV_decoder, ls_t_prior is required",
         call. = FALSE)
  }
  if (inherits(decoder_time, "deepRV_decoder")) {
    validate_prior_in_range(ls_t_prior, decoder_time$ls_trained_range,
                            decoder_label = sprintf("<time decoder %s>",
                                                    decoder_time$kernel))
  }
  fs <- family_spec(family, by_mode = "none")
  fs$likelihood <- sub("mu\\[obs_idx\\]", "spatial", fs$likelihood)
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)

  tb <- time_block(decoder_time, ls_t_prior)
  sigma_t_line <- if (tb$include_sigma_t) "  real<lower=0> sigma_t;" else NULL
  sigma_t_prior_line <- if (tb$include_sigma_t)
    stan_prior_statement(sigma_t_prior, "sigma_t") else NULL

  # Extra data declarations: time-decoder weights if decoder_time is a decoder.
  time_data <- if (inherits(decoder_time, "deepRV_decoder")) c(
    "  int<lower=1> cond_dim_t;",
    "  int<lower=1> hidden_t;",
    "  matrix[T + cond_dim_t, hidden_t] W1_t;",
    "  vector[hidden_t]                 b1_t;",
    "  matrix[hidden_t, T]              W2_t;",
    "  vector[T]                        b2_t;"
  ) else NULL

  lines <- c(
    "functions {",
    decode_body,
    "}",
    "data {",
    "  int<lower=1> N;",
    "  int<lower=1> L;",            # = N_side^2
    "  int<lower=1> N_side;",
    "  int<lower=1> T;",
    "  int<lower=1> cond_dim;",
    "  int<lower=1> hidden;",
    "  int<lower=1> K;",
    "  matrix[N, K] X;",
    "  matrix[N_side + cond_dim, hidden] W1;",   # axis decoder
    "  vector[hidden]                    b1;",
    "  matrix[hidden, N_side]            W2;",
    "  vector[N_side]                    b2;",
    time_data,
    fs$y_decl,
    "  array[N] int<lower=1, upper=L>  obs_idx;",
    "  array[N] int<lower=1, upper=T>  time_idx;",
    "}",
    "parameters {",
    "  matrix[T, L] z;",
    sprintf("  real<lower=%s, upper=%s> ls_x;", ls_lo, ls_hi),
    sprintf("  real<lower=%s, upper=%s> ls_y;", ls_lo, ls_hi),
    sigma_t_line,
    tb$pars,
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    "  vector[cond_dim] cond_x;",
    "  vector[cond_dim] cond_y;",
    "  cond_x[1] = ls_x;",
    "  cond_y[1] = ls_y;",
    # Per time step Kronecker spatial decode.
    "  matrix[T, L] eps;",
    "  for (t in 1:T) {",
    "    matrix[N_side, N_side] zt = to_matrix(to_vector(z[t]), N_side, N_side);",
    "    matrix[N_side, N_side] mid;",
    "    for (j in 1:N_side)",
    "      mid[, j] = decode(zt[, j], cond_x, W1, b1, W2, b2);",
    "    matrix[N_side, N_side] mu_grid;",
    "    for (i in 1:N_side)",
    "      mu_grid[i] = decode(to_vector(mid[i]), cond_y, W1, b1, W2, b2)';",
    "    eps[t] = to_vector(mu_grid)';",
    "  }",
    tb$F_block_post_eps,
    "}",
    "model {",
    "  to_vector(z) ~ std_normal();",
    "  b ~ std_normal();",
    fs$extra_prior,
    sigma_t_prior_line,
    tb$priors,
    sprintf("  // ls_x, ls_y ~ uniform(%s, %s)  -- implicit from parameter bounds",
            ls_lo, ls_hi),
    "  vector[N] spatial;",
    "  for (n in 1:N) spatial[n] = F[time_idx[n], obs_idx[n]];",
    fs$likelihood,
    "}"
  )
  paste(Filter(Negate(is.null), lines), collapse = "\n")
}

build_stancode <- function(decoder, ls_prior, family, by_mode = "none",
                           ls_pooling = "complete") {
  if (inherits(decoder, "deepRV_decoder_kron")) {
    if (by_mode != "none") {
      stop("by != NULL is not supported with Kronecker decoders in v0.1",
           call. = FALSE)
    }
    if (ls_pooling != "complete") {
      stop("ls_pooling != \"complete\" is not supported with Kronecker ",
           "decoders in v0.1.", call. = FALSE)
    }
    return(build_stancode_kron(decoder, ls_prior, family))
  }
  build_stancode_1d(decoder, ls_prior, family, by_mode, ls_pooling)
}

# Standata for the space-time term. Same shape as 1D + adds T and
# time_idx; obs_idx remains the spatial index.
build_standata_st <- function(decoder, deepRV_st_call, y, X, family) {
  obs_idx <- as.integer(deepRV_st_call$obs_idx)
  time_idx <- as.integer(deepRV_st_call$time_idx)
  if (length(obs_idx) != length(y) || length(time_idx) != length(y)) {
    stop(sprintf(
      "obs_idx (len %d), time_idx (len %d), and y (len %d) must all match",
      length(obs_idx), length(time_idx), length(y)),
      call. = FALSE)
  }
  if (any(is.na(obs_idx)) || any(obs_idx < 1L) || any(obs_idx > decoder$L)) {
    stop(sprintf("obs_idx values must be integers in [1, L=%d]", decoder$L),
         call. = FALSE)
  }
  T_full <- max(time_idx, na.rm = TRUE)
  if (any(is.na(time_idx)) || any(time_idx < 1L)) {
    stop("time_idx must be positive integers", call. = FALSE)
  }
  if (!is.matrix(X) || nrow(X) != length(y)) {
    stop("design matrix X must have nrow == length(y)", call. = FALSE)
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
      as.numeric(y)
    },
    stop(sprintf("unsupported family: %s", family), call. = FALSE)
  )
  is_kron <- inherits(decoder, "deepRV_decoder_kron")
  # For Kron spatial: ship the axis-decoder weights (axis decoder feeds
  # both x and y) plus N_side. For 1D: ship the full decoder weights.
  weights_src <- if (is_kron) decoder$x_decoder else decoder
  out <- list(
    N        = length(y),
    L        = as.integer(decoder$L),
    T        = as.integer(T_full),
    cond_dim = length(weights_src$conditionals),
    hidden   = ncol(weights_src$weights$W1),
    K        = ncol(X),
    X        = X,
    W1       = weights_src$weights$W1,
    b1       = as.array(weights_src$weights$b1),
    W2       = weights_src$weights$W2,
    b2       = as.array(weights_src$weights$b2),
    y        = y_data,
    obs_idx  = obs_idx,
    time_idx = time_idx
  )
  if (is_kron) out$N_side <- as.integer(decoder$grid_side)
  dec_t <- deepRV_st_call$decoder_time
  if (inherits(dec_t, "deepRV_decoder")) {
    if (as.integer(dec_t$L) != as.integer(T_full)) {
      stop(sprintf(
        "time decoder grid size (%d) must equal the observed T (%d)",
        dec_t$L, T_full), call. = FALSE)
    }
    out$cond_dim_t <- length(dec_t$conditionals)
    out$hidden_t   <- ncol(dec_t$weights$W1)
    out$W1_t       <- dec_t$weights$W1
    out$b1_t       <- as.array(dec_t$weights$b1)
    out$W2_t       <- dec_t$weights$W2
    out$b2_t       <- as.array(dec_t$weights$b2)
  }
  out
}

# Build the 1D Stan program. ls_prior may be NULL for the (future)
# case where ls has no user prior; for now prior_uniform is required.
#
# by_mode controls how the decoder's mu contributes to eta:
#   "none"   : eta_n = X[n,:] %*% b + mu[obs_idx[n]]                  (default)
#   "svc"    : eta_n = X[n,:] %*% b + x_by[n] * mu[obs_idx[n]]        (SVC)
#   "factor" : matrix[G, L] z; matrix[G, L] mu; eta_n adds
#              mu[group_idx[n], obs_idx[n]].
#
# ls_pooling applies only when by_mode = "factor":
#   "complete" : real ls (single, shared)
#   "none"     : vector[G] ls (independent, each gets the same uniform prior)
#   "partial"  : vector[G] ls drawn from N(mu_ls, tau_ls^2) truncated to the
#                trained range; mu_ls inherits the ls_prior bounds, tau_ls has
#                a half-normal(0, 0.5) default.
build_stancode_1d <- function(decoder, ls_prior, family, by_mode = "none",
                              ls_pooling = "complete") {
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")
  stopifnot(by_mode %in% c("none", "svc", "factor"))
  stopifnot(ls_pooling %in% c("complete", "none", "partial"))
  if (by_mode != "factor" && ls_pooling != "complete") {
    stop("ls_pooling is only meaningful with by = <factor>; ",
         "non-default ls_pooling values are rejected otherwise",
         call. = FALSE)
  }
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
    # ls is scalar for "complete", vector[G] otherwise.
    ls_par <- switch(ls_pooling,
      complete = sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi),
      none     = sprintf("  vector<lower=%s, upper=%s>[G] ls;", ls_lo, ls_hi),
      partial  = c(
        sprintf("  vector<lower=%s, upper=%s>[G] ls;", ls_lo, ls_hi),
        sprintf("  real<lower=%s, upper=%s> mu_ls;", ls_lo, ls_hi),
        "  real<lower=0> tau_ls;"
      )
    )
    # Decode loop. For ls_pooling != "complete" the conditional changes
    # per group, so we set cond[1] inside the loop.
    if (ls_pooling == "complete") {
      mu_block <- c(
        "  vector[cond_dim] cond;",
        "  cond[1] = ls;",
        "  matrix[G, L] mu;",
        "  for (g in 1:G)",
        "    mu[g] = decode(to_vector(z[g]), cond, W1, b1, W2, b2)';"
      )
    } else {
      mu_block <- c(
        "  matrix[G, L] mu;",
        "  for (g in 1:G) {",
        "    vector[cond_dim] cond_g;",
        "    cond_g[1] = ls[g];",
        "    mu[g] = decode(to_vector(z[g]), cond_g, W1, b1, W2, b2)';",
        "  }"
      )
    }
    extra_ls_prior <- switch(ls_pooling,
      complete = NULL,
      none     = NULL,    # implicit uniform from element-wise bounds
      partial  = c(
        "  ls ~ normal(mu_ls, tau_ls);  // truncated by element-wise bounds",
        "  tau_ls ~ normal(0, 0.5);     // default half-normal(0, 0.5)"
      )
    )
    pre_likelihood <- c(
      "  vector[N] spatial;",
      "  for (n in 1:N) spatial[n] = mu[group_idx[n], obs_idx[n]];"
    )
  } else {
    z_par <- "  vector[L] z;"
    z_prior <- "  z ~ std_normal();"
    ls_par <- sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi)
    mu_block <- c(
      "  vector[cond_dim] cond;",
      "  cond[1] = ls;",
      "  vector[L] mu = decode(z, cond, W1, b1, W2, b2);"
    )
    extra_ls_prior <- NULL
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
    ls_par,
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
    extra_ls_prior,
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
