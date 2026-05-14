# Build the full Stan source for an MVP deepRV model. The MVP is
# constrained to:
#   - exactly one deepRV() term
#   - intercept-only fixed effects (no covariates)
#   - Poisson family
#   - prior_uniform on ls
# Each restriction lifts in a follow-on commit; the check is in
# deeprv_brm() so this codegen can be reused.
#
# The decoder's `decode` function source is loaded from
# inst/stan/decode_mlp.stan and inlined verbatim - keeping that file
# the single source of truth for the forward pass.

read_decode_functions_block <- function() {
  path <- system.file("stan", "decode_mlp.stan", package = "brms.deeprv")
  if (!nzchar(path) || !file.exists(path)) {
    stop("inst/stan/decode_mlp.stan not found in the installed package",
         call. = FALSE)
  }
  src <- paste(readLines(path, warn = FALSE), collapse = "\n")
  # decode_mlp.stan has a single `functions { ... }` block plus header
  # comments. Strip the wrapping braces so we can splice into the body
  # of a larger `functions { }` block in the generated program.
  m <- regmatches(src, regexpr("functions\\s*\\{[\\s\\S]*\\}", src, perl = TRUE))
  if (length(m) != 1L) {
    stop("could not locate `functions { ... }` block in decode_mlp.stan",
         call. = FALSE)
  }
  # Drop the leading "functions {" and trailing "}".
  inner <- sub("^functions\\s*\\{", "", m)
  inner <- sub("\\}\\s*$", "", inner)
  trimws(inner)
}

# Build the Stan program string. The caller is responsible for passing
# a validated decoder + prior; we just stitch the templates here.
build_stancode <- function(decoder, ls_prior, family) {
  stopifnot(family == "poisson")
  stopifnot(inherits(ls_prior, "deepRV_prior"))
  stopifnot(ls_prior$family == "uniform")

  decode_body <- read_decode_functions_block()
  ls_lo <- sprintf("%.10g", ls_prior$lower)
  ls_hi <- sprintf("%.10g", ls_prior$upper)

  sprintf(
    paste(
      "functions {",
      "%s",
      "}",
      "data {",
      "  int<lower=1> N;",
      "  int<lower=1> L;",
      "  int<lower=1> cond_dim;",
      "  int<lower=1> hidden;",
      "  matrix[L + cond_dim, hidden] W1;",
      "  vector[hidden]               b1;",
      "  matrix[hidden, L]            W2;",
      "  vector[L]                    b2;",
      "  array[N] int<lower=0>           y;",
      "  array[N] int<lower=1, upper=L>  obs_idx;",
      "}",
      "parameters {",
      "  vector[L] z;",
      "  real<lower=%s, upper=%s> ls;",
      "  real beta0;",
      "}",
      "transformed parameters {",
      "  vector[cond_dim] cond;",
      "  cond[1] = ls;",
      "  vector[L] mu = decode(z, cond, W1, b1, W2, b2);",
      "}",
      "model {",
      "  z ~ std_normal();",
      "  beta0 ~ std_normal();",
      "  // ls ~ uniform(%s, %s)  -- implicit from parameter bounds",
      "  target += poisson_log_lpmf(y | beta0 + mu[obs_idx]);",
      "}",
      sep = "\n"
    ),
    decode_body, ls_lo, ls_hi, ls_lo, ls_hi
  )
}

# Build the Stan data list from a decoder + the parsed deepRV term +
# the response vector y. Aborts on any shape mismatch.
build_standata <- function(decoder, deepRV_call, y) {
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
  if (!is.numeric(y) || any(y < 0) || any(y != round(y))) {
    stop("Poisson family requires non-negative integer y", call. = FALSE)
  }
  list(
    N        = length(y),
    L        = as.integer(decoder$L),
    cond_dim = length(decoder$conditionals),
    hidden   = ncol(decoder$weights$W1),
    W1       = decoder$weights$W1,
    b1       = as.array(decoder$weights$b1),
    W2       = decoder$weights$W2,
    b2       = as.array(decoder$weights$b2),
    y        = as.integer(y),
    obs_idx  = obs_idx
  )
}
