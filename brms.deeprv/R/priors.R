#' Uniform prior over the decoder's length-scale parameter
#'
#' Returned object is consumed by [deeprv_brm()] and rendered into the
#' generated Stan program as a parameter bound. Must lie inside the
#' decoder's `ls_trained_range`; [deeprv_brm()] errors otherwise (see
#' `DESIGN.md` section 2.3.3).
#'
#' @param lower,upper Bounds. `lower < upper`, both finite, both > 0.
#' @return A `deepRV_prior` object.
#' @export
prior_uniform <- function(lower, upper) {
  if (!is.numeric(lower) || length(lower) != 1L ||
      !is.numeric(upper) || length(upper) != 1L) {
    stop("`lower` and `upper` must be single numeric values", call. = FALSE)
  }
  if (!is.finite(lower) || !is.finite(upper) || lower <= 0 || lower >= upper) {
    stop(sprintf("invalid uniform prior: lower=%g, upper=%g", lower, upper),
         call. = FALSE)
  }
  out <- list(family = "uniform", lower = as.numeric(lower),
              upper = as.numeric(upper))
  class(out) <- c("deepRV_prior", "list")
  out
}

#' @export
print.deepRV_prior <- function(x, ...) {
  cat(sprintf("<deepRV_prior: %s(lower=%g, upper=%g)>\n",
              x$family, x$lower, x$upper))
  invisible(x)
}

# Lower bound of a prior's support, used for Stan parameter bounds.
prior_support_lower <- function(prior) {
  switch(prior$family,
    uniform = prior$lower,
    stop(sprintf("unsupported prior family: %s", prior$family), call. = FALSE)
  )
}

prior_support_upper <- function(prior) {
  switch(prior$family,
    uniform = prior$upper,
    stop(sprintf("unsupported prior family: %s", prior$family), call. = FALSE)
  )
}

# Emit a Stan statement that adds the prior contribution to `target`. For
# a uniform prior the parameter bounds carry the entire density, so the
# emitted line is just a comment.
stan_prior_statement <- function(prior, param_name) {
  switch(prior$family,
    uniform = sprintf("  // %s ~ uniform(%g, %g) - implicit from bounds",
                      param_name, prior$lower, prior$upper),
    stop(sprintf("unsupported prior family: %s", prior$family), call. = FALSE)
  )
}

# Verify a user's prior is a subset of the decoder's trained range.
# DESIGN.md section 2.3.3: hard error if not.
validate_prior_in_range <- function(prior, ls_trained_range,
                                    decoder_label = "<decoder>") {
  lo <- prior_support_lower(prior)
  hi <- prior_support_upper(prior)
  trained_lo <- ls_trained_range[1]
  trained_hi <- ls_trained_range[2]
  if (lo < trained_lo || hi > trained_hi) {
    stop(sprintf(
      paste0("ls prior support [%g, %g] extends outside %s's trained range ",
             "[%g, %g]. The decoder is not guaranteed to be accurate outside ",
             "the training range; pick a narrower prior or load a decoder ",
             "trained on a wider range."),
      lo, hi, decoder_label, trained_lo, trained_hi),
      call. = FALSE)
  }
  invisible(TRUE)
}
