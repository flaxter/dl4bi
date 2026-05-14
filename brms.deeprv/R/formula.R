#' Mark a deepRV decoder term in a [deeprv_brm()] formula
#'
#' `deepRV()` doesn't compute anything on its own - it captures its
#' arguments so [deeprv_brm()] can recognise the term during formula
#' parsing and translate it into the appropriate Stan additions.
#'
#' @param s Coordinate column from `data`. Used for diagnostics only
#'   in v0.1; `obs_idx` is what actually drives indexing into the
#'   decoder's grid.
#' @param decoder A `deepRV_decoder` from [load_deeprv()]. Required.
#' @param obs_idx Integer vector of length `nrow(data)` mapping each
#'   observation to its 1-based grid index in `decoder$grid_coords`.
#'   Required.
#' @param ls_prior A `deepRV_prior` from [prior_uniform()]. Required.
#'   Must lie inside `decoder$ls_trained_range`.
#' @param by,gr,ls_pooling Reserved for v0.1 follow-on features. Must
#'   be at their defaults in the MVP; non-default values error.
#' @export
deepRV <- function(s, decoder, obs_idx, ls_prior,
                   by = NULL, gr = FALSE, ls_pooling = "complete") {
  call_args <- list(
    s_expr     = substitute(s),
    decoder    = decoder,
    obs_idx    = obs_idx,
    ls_prior   = ls_prior,
    by         = by,
    gr         = gr,
    ls_pooling = ls_pooling
  )
  class(call_args) <- c("deepRV_call", "list")
  call_args
}

#' @export
print.deepRV_call <- function(x, ...) {
  cat("<deepRV_call>\n")
  cat(sprintf("  s        : %s\n", deparse(x$s_expr)))
  cat(sprintf("  decoder  : %s grid=%d kernel=%s\n",
              x$decoder$domain, x$decoder$grid_size, x$decoder$kernel))
  cat(sprintf("  obs_idx  : length %d\n", length(x$obs_idx)))
  cat(sprintf("  ls_prior : %s(%g, %g)\n",
              x$ls_prior$family, x$ls_prior$lower, x$ls_prior$upper))
  invisible(x)
}

# Walk the RHS of a formula collecting every deepRV(...) call. Returns
# a list with two slots:
#   $deepRV : a list of evaluated deepRV_call objects (one per term)
#   $rhs    : the residual RHS expression with deepRV(...) terms removed,
#             or NULL if every term was a deepRV() call.
#
# Evaluation happens in the caller's environment plus `data` so the
# arguments to deepRV() can reference symbols like `s` and `obs_idx`
# from the data frame. This matches how brms::gp() resolves its args.
parse_deeprv_formula <- function(formula, data, envir = parent.frame()) {
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula", call. = FALSE)
  }
  if (length(formula) != 3L) {
    stop("`formula` must have both an LHS and an RHS", call. = FALSE)
  }
  rhs <- formula[[3L]]
  eval_env <- list2env(as.list(data), parent = envir)
  collected <- list()

  # Recursive walk: split by `+` (and `-`), evaluating deepRV() calls.
  # Everything that isn't a deepRV() call is retained in the residual RHS.
  walk <- function(e) {
    if (is.call(e) && length(e) >= 1L) {
      head <- e[[1L]]
      if (is.name(head) && identical(as.character(head), "deepRV")) {
        # Resolve to brms.deeprv::deepRV so that re-named imports still work.
        call_eval <- e
        call_eval[[1L]] <- quote(brms.deeprv::deepRV)
        spec <- eval(call_eval, envir = eval_env)
        if (!inherits(spec, "deepRV_call")) {
          stop("`deepRV(...)` did not produce a deepRV_call object",
               call. = FALSE)
        }
        collected[[length(collected) + 1L]] <<- spec
        return(NULL)
      }
      if (is.name(head) && identical(as.character(head), "+")) {
        lhs <- walk(e[[2L]])
        rhs_inner <- walk(e[[3L]])
        if (is.null(lhs))      return(rhs_inner)
        if (is.null(rhs_inner)) return(lhs)
        return(call("+", lhs, rhs_inner))
      }
    }
    e
  }
  residual <- walk(rhs)
  if (length(collected) == 0L) {
    stop("formula has no deepRV() term. Use brms::brm() directly for ",
         "non-deepRV models.", call. = FALSE)
  }
  if (length(collected) > 1L) {
    stop(sprintf("multiple deepRV() terms in formula (%d); v0.1 supports ",
                 length(collected)),
         "exactly one. Stacking multiple decoders is a planned v0.2 feature.",
         call. = FALSE)
  }
  list(deepRV = collected, rhs = residual, lhs = formula[[2L]])
}
