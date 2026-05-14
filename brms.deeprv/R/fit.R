#' Fit a deepRV-augmented brms-style model
#'
#' MVP scope:
#' \itemize{
#'   \item Exactly one `deepRV()` term in the formula.
#'   \item Intercept-only fixed effects (no covariates other than the
#'     decoder term).
#'   \item Poisson family.
#'   \item `prior_uniform()` on the length-scale.
#' }
#' Each of these restrictions is checked at the top of `deeprv_brm()`
#' and lifts in a follow-on. The wider `by` / `gr` / `ls_pooling`
#' / fixed-effects / family surface is documented in DESIGN.md sections
#' 2.3 and 2.4.
#'
#' @param formula A formula of the form `y ~ deepRV(s, decoder = ...,
#'   obs_idx = ..., ls_prior = ...)`.
#' @param data A data frame containing the response and any columns
#'   referenced inside `deepRV(...)`.
#' @param family A family object. Only `poisson()` is supported in MVP.
#' @param chains,iter,warmup,seed,cores,control Forwarded to
#'   [rstan::sampling()].
#' @param ... Caught and rejected for now (no brms passthrough yet).
#'
#' @return A `deeprv_fit` object containing the `stanfit`, the
#'   generated Stan code, the standata, and the parsed `deepRV_call`.
#'
#' @export
deeprv_brm <- function(formula, data, family = stats::poisson(),
                       chains = 2L, iter = 1000L, warmup = floor(iter / 2),
                       seed = NULL, cores = getOption("mc.cores", 1L),
                       control = list(adapt_delta = 0.95),
                       ...) {
  extra <- list(...)
  if (length(extra) > 0L) {
    stop("MVP deeprv_brm() doesn't accept additional arguments yet. ",
         sprintf("Got: %s", paste(names(extra), collapse = ", ")),
         call. = FALSE)
  }
  if (!requireNamespace("rstan", quietly = TRUE)) {
    stop("`rstan` is required for deeprv_brm() but is not installed",
         call. = FALSE)
  }
  fam <- as_family(family)
  if (!identical(fam$family, "poisson")) {
    stop("MVP supports family = poisson() only; got: ",
         fam$family, call. = FALSE)
  }

  parsed <- parse_deeprv_formula(formula, data, envir = parent.frame())
  dr_call <- parsed$deepRV[[1L]]
  if (!is.null(parsed$rhs)) {
    stop("MVP only supports an intercept and one deepRV() term. ",
         "Additional fixed effects on the RHS aren't wired up yet: ",
         deparse(parsed$rhs), call. = FALSE)
  }
  # Reject the v0.2 knobs explicitly so users get a clear error.
  if (!is.null(dr_call$by)) {
    stop("MVP: deepRV(by = ...) is reserved for a follow-up. Drop the ",
         "`by` argument to fit a single shared field.", call. = FALSE)
  }
  if (!isFALSE(dr_call$gr)) {
    stop("MVP: deepRV(gr = TRUE) is reserved for a follow-up", call. = FALSE)
  }
  if (!identical(dr_call$ls_pooling, "complete")) {
    stop("MVP: deepRV(ls_pooling = ...) only supports the default \"complete\"",
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

  stancode <- build_stancode(decoder, dr_call$ls_prior, family = "poisson")
  standata <- build_standata(decoder, dr_call, y)

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
    stanfit  = stanfit,
    stancode = stancode,
    standata = standata,
    decoder  = decoder,
    deepRV   = dr_call,
    family   = fam
  )
  class(out) <- c("deeprv_fit", "list")
  out
}

#' @export
print.deeprv_fit <- function(x, ...) {
  cat("<deeprv_fit>\n")
  cat(sprintf("  decoder : %s grid=%d kernel=%s\n",
              x$decoder$domain, x$decoder$grid_size, x$decoder$kernel))
  cat(sprintf("  family  : %s\n", x$family$family))
  cat(sprintf("  N       : %d   L : %d\n",
              x$standata$N, x$standata$L))
  cat("  stanfit summary (beta0, ls):\n")
  print(x$stanfit, pars = c("beta0", "ls"), probs = c(0.05, 0.5, 0.95))
  invisible(x)
}

# Normalize family inputs: accept the family object, a string, or a
# function returning a family.
as_family <- function(family) {
  if (inherits(family, "family")) return(family)
  if (is.function(family))        return(family())
  if (is.character(family))       return(get(family, mode = "function")())
  stop("`family` must be a family object, function, or string", call. = FALSE)
}

decoder_label <- function(dr) {
  sprintf("%s/grid_size=%d/kernel=%s", dr$domain, dr$grid_size, dr$kernel)
}
