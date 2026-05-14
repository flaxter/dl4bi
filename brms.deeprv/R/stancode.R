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
#   "none" -> mu[obs_idx]
#   "svc"  -> x_by .* mu[obs_idx]    (element-wise SVC)
family_spec <- function(family_name, by_mode = "none") {
  spatial_term <- switch(by_mode,
    none = "mu[obs_idx]",
    svc  = "x_by .* mu[obs_idx]",
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

# Build the full Stan program. ls_prior may be NULL for the (future)
# case where ls has no user prior; for now prior_uniform is required.
#
# by_mode controls how the decoder's mu contributes to eta:
#   "none" : eta_n = X[n,:] %*% b + mu[obs_idx[n]]                  (default)
#   "svc"  : eta_n = X[n,:] %*% b + x_by[n] * mu[obs_idx[n]]        (SVC)
build_stancode <- function(decoder, ls_prior, family, by_mode = "none") {
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")
  stopifnot(by_mode %in% c("none", "svc"))
  fs <- family_spec(family, by_mode)
  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)
  extra_data <- if (by_mode == "svc") "  vector[N] x_by;" else NULL

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
    "  vector[L] z;",
    sprintf("  real<lower=%s, upper=%s> ls;", ls_lo, ls_hi),
    "  vector[K] b;",
    fs$extra_par,
    "}",
    "transformed parameters {",
    "  vector[cond_dim] cond;",
    "  cond[1] = ls;",
    "  vector[L] mu = decode(z, cond, W1, b1, W2, b2);",
    "}",
    "model {",
    "  z ~ std_normal();",
    "  b ~ std_normal();",
    fs$extra_prior,
    sprintf("  // ls ~ uniform(%s, %s)  -- implicit from parameter bounds",
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
#   - by_values for SVC (passed in as `x_by`)
build_standata <- function(decoder, deepRV_call, y, X, family,
                           by_mode = "none", by_values = NULL) {
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
  if (by_mode == "svc") {
    if (is.null(by_values) || length(by_values) != length(y)) {
      stop("by_values must have the same length as y for SVC mode",
           call. = FALSE)
    }
    out$x_by <- as.numeric(by_values)
  }
  out
}
