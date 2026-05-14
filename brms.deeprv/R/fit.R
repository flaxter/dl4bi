#' Fit a deepRV-augmented brms-style model
#'
#' Supported v0.1 surface:
#' \itemize{
#'   \item Exactly one `deepRV()` term in the formula.
#'   \item Any RHS covariates handled via `model.matrix(~ rhs, data)`.
#'     Coefficient vector `b` gets a default N(0, 1) prior.
#'   \item `family` in `poisson()` or `gaussian()`.
#'   \item `prior_uniform()` on the length-scale.
#' }
#' The `by` / `gr` / `ls_pooling` knobs are reserved for v0.2 and
#' error if non-default.
#'
#' @param formula A formula of the form `y ~ <covariates> +
#'   deepRV(s, decoder = ..., obs_idx = ..., ls_prior = ...)`.
#' @param data A data frame containing the response, any RHS columns,
#'   and any columns referenced inside `deepRV(...)`.
#' @param family A family object. `poisson()` and `gaussian()` are
#'   supported.
#' @param chains,iter,warmup,seed,cores,control Forwarded to
#'   `rstan::stan()`.
#' @param ... Caught and rejected for now (no brms passthrough yet).
#'
#' @return A `deeprv_fit` object containing the `stanfit`, the
#'   generated Stan code, the standata, the parsed `deepRV_call`,
#'   and the design matrix `X` for downstream prediction.
#'
#' @export
deeprv_brm <- function(formula, data, family = stats::poisson(),
                       chains = 2L, iter = 1000L, warmup = floor(iter / 2),
                       seed = NULL, cores = getOption("mc.cores", 1L),
                       control = list(adapt_delta = 0.95),
                       ...) {
  extra <- list(...)
  if (length(extra) > 0L) {
    stop("deeprv_brm() doesn't accept additional arguments yet. ",
         sprintf("Got: %s", paste(names(extra), collapse = ", ")),
         call. = FALSE)
  }
  if (!requireNamespace("rstan", quietly = TRUE)) {
    stop("`rstan` is required for deeprv_brm() but is not installed",
         call. = FALSE)
  }
  fam <- as_family(family)
  if (!(fam$family %in% c("poisson", "gaussian"))) {
    stop("supported families: poisson, gaussian; got: ", fam$family,
         call. = FALSE)
  }

  parsed <- parse_deeprv_formula(formula, data, envir = parent.frame())
  dr_call <- parsed$deepRV[[1L]]
  by_info <- classify_by(dr_call$by)
  if (!isFALSE(dr_call$gr)) {
    stop("deepRV(gr = TRUE) is reserved for a follow-up", call. = FALSE)
  }
  if (!identical(dr_call$ls_pooling, "complete")) {
    stop("deepRV(ls_pooling = ...) only supports the default \"complete\"",
         call. = FALSE)
  }

  decoder <- dr_call$decoder
  if (!inherits(decoder, "deepRV_decoder")) {
    stop("`decoder` must be a deepRV_decoder from load_deeprv()",
         call. = FALSE)
  }
  validate_decoder(decoder, source = "<deeprv_brm input>")
  validate_prior_in_range(dr_call$ls_prior, decoder$ls_trained_range,
                          decoder_label = decoder_label(decoder))

  y <- eval(parsed$lhs, envir = data, enclos = parent.frame())

  X <- build_design_matrix(parsed$rhs, data)

  if (by_info$mode %in% c("svc", "factor") &&
      length(by_info$values) != length(y)) {
    stop(sprintf(
      "deepRV(by = ...) vector length (%d) must match nrow(data) (%d)",
      length(by_info$values), length(y)),
      call. = FALSE)
  }

  stancode <- build_stancode(decoder, dr_call$ls_prior, family = fam$family,
                             by_mode = by_info$mode)
  standata <- build_standata(decoder, dr_call, y, X, family = fam$family,
                             by_mode = by_info$mode,
                             by_values = by_info$values,
                             n_groups = by_info$n_groups)

  sampling_args <- list(
    model_code = stancode,
    data       = standata,
    chains     = chains,
    iter       = iter,
    warmup     = warmup,
    cores      = cores,
    control    = control,
    refresh    = 0
  )
  if (!is.null(seed)) sampling_args$seed <- seed
  stanfit <- do.call(rstan::stan, sampling_args)

  out <- list(
    stanfit   = stanfit,
    stancode  = stancode,
    standata  = standata,
    decoder   = decoder,
    deepRV    = dr_call,
    family    = fam,
    X         = X,
    by_mode   = by_info$mode,
    by_values = by_info$values,
    by_levels = by_info$levels,
    n_groups  = by_info$n_groups
  )
  class(out) <- c("deeprv_fit", "list")
  out
}

# Classify the deepRV(..., by = ...) value:
#   NULL           -> mode = "none"   (single shared field)
#   numeric vector -> mode = "svc"    (spatially varying coefficient)
#   factor / chr   -> mode = "factor" (G group-specific smooths)
#
# For factor mode, normalises via factor() so character vectors and
# already-factor inputs produce the same integer group ids + level names.
classify_by <- function(by_val) {
  if (is.null(by_val)) {
    return(list(mode = "none", values = NULL))
  }
  if (is.factor(by_val) || is.character(by_val)) {
    f <- factor(by_val)
    return(list(
      mode      = "factor",
      values    = as.integer(f),
      n_groups  = length(levels(f)),
      levels    = levels(f)
    ))
  }
  if (is.numeric(by_val)) {
    return(list(mode = "svc", values = as.numeric(by_val)))
  }
  stop("deepRV(by = ...) must be NULL, a numeric vector, or a factor",
       call. = FALSE)
}

#' @export
print.deeprv_fit <- function(x, ...) {
  cat("<deeprv_fit>\n")
  cat(sprintf("  decoder : %s grid=%d kernel=%s\n",
              x$decoder$domain, x$decoder$grid_size, x$decoder$kernel))
  cat(sprintf("  family  : %s\n", x$family$family))
  cat(sprintf("  N       : %d   L : %d   K : %d\n",
              x$standata$N, x$standata$L, x$standata$K))
  pars <- c("b", "ls")
  if (x$family$family == "gaussian") pars <- c(pars, "sigma")
  cat("  stanfit summary:\n")
  print(x$stanfit, pars = pars, probs = c(0.05, 0.5, 0.95))
  invisible(x)
}

as_family <- function(family) {
  if (inherits(family, "family")) return(family)
  if (is.function(family))        return(family())
  if (is.character(family))       return(get(family, mode = "function")())
  stop("`family` must be a family object, function, or string", call. = FALSE)
}

decoder_label <- function(dr) {
  sprintf("%s/grid_size=%d/kernel=%s", dr$domain, dr$grid_size, dr$kernel)
}

# Turn the residual RHS of a formula into a numeric design matrix. If
# the residual is NULL (no covariates beyond the deepRV() term), defaults
# to an intercept-only matrix of 1's.
#
# Uses model.matrix's standard rules: an intercept is included by default
# unless the user wrote `... - 1` or `... + 0`.
build_design_matrix <- function(rhs_expr, data) {
  if (is.null(rhs_expr)) {
    rhs_formula <- ~1
  } else {
    rhs_formula <- stats::reformulate(deparse(rhs_expr, width.cutoff = 500L))
  }
  mf <- stats::model.frame(rhs_formula, data = data, na.action = stats::na.fail)
  X <- stats::model.matrix(rhs_formula, mf)
  # Drop the model.matrix dimnames since Stan doesn't care; keep colnames
  # for downstream posterior_predict but not the row names (memory).
  rownames(X) <- NULL
  X
}
