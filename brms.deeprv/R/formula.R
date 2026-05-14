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

#' Mark a deepRV space-time term in a [deeprv_brm()] formula
#'
#' Like [deepRV()] but with an additional time axis. Currently MVP scope:
#' 1D `unit_interval` decoder for space + a random-walk time prior. The
#' design doc allows `decoder_time` to also be `"ar1"` or a 1D decoder;
#' those will land in a follow-up.
#'
#' @param decoder A 1D `deepRV_decoder` for the spatial axis (from
#'   [load_deeprv()]).
#' @param decoder_time One of:
#'   \itemize{
#'     \item `"rw"`: random walk in time. Each time slice's spatial
#'       field is added to the previous, scaled by `sigma_t`.
#'   }
#' @param obs_idx Integer vector of length `nrow(data)` mapping each
#'   observation to its 1-based spatial grid index.
#' @param time_idx Integer vector of length `nrow(data)` mapping each
#'   observation to its 1-based time index.
#' @param ls_prior A `deepRV_prior` for the spatial length-scale.
#' @param sigma_t_prior A `deepRV_prior` for the random-walk innovation
#'   scale. Default `prior_exp(1)`.
#' @export
deepRV_st <- function(decoder, decoder_time = "rw",
                      obs_idx, time_idx, ls_prior,
                      sigma_t_prior = prior_exp(1),
                      ls_t_prior = NULL) {
  if (!inherits(decoder, "deepRV_decoder")) {
    stop("`decoder` must be a deepRV_decoder from load_deeprv()",
         call. = FALSE)
  }
  if (inherits(decoder, "deepRV_decoder_kron")) {
    # The Kron spatial path is handled in build_stancode_st_kron; the
    # `decoder_time` argument still applies as one of rw / ar1 / decoder.
  }
  if (inherits(decoder_time, "deepRV_decoder")) {
    if (is.null(ls_t_prior)) {
      stop("when decoder_time is a deepRV_decoder, ls_t_prior is required",
           call. = FALSE)
    }
    if (!inherits(ls_t_prior, "deepRV_prior")) {
      stop("ls_t_prior must be a deepRV_prior (e.g. prior_uniform(...))",
           call. = FALSE)
    }
  } else if (is.character(decoder_time) && length(decoder_time) == 1L &&
             decoder_time %in% c("rw", "ar1")) {
    # OK
  } else {
    stop("deepRV_st(): decoder_time must be \"rw\", \"ar1\", or a ",
         "deepRV_decoder for time", call. = FALSE)
  }
  call_args <- list(
    decoder       = decoder,
    decoder_time  = decoder_time,
    obs_idx       = obs_idx,
    time_idx      = time_idx,
    ls_prior      = ls_prior,
    sigma_t_prior = sigma_t_prior,
    ls_t_prior    = ls_t_prior
  )
  class(call_args) <- c("deepRV_st_call", "list")
  call_args
}

#' @export
print.deepRV_st_call <- function(x, ...) {
  cat("<deepRV_st_call>\n")
  cat(sprintf("  decoder       : %s grid=%d kernel=%s\n",
              x$decoder$domain, x$decoder$grid_size, x$decoder$kernel))
  cat(sprintf("  decoder_time  : %s\n", x$decoder_time))
  cat(sprintf("  obs_idx       : length %d\n", length(x$obs_idx)))
  cat(sprintf("  time_idx      : length %d, T = %d\n",
              length(x$time_idx), max(x$time_idx, na.rm = TRUE)))
  cat("  ls_prior      : "); print(x$ls_prior)
  cat("  sigma_t_prior : "); print(x$sigma_t_prior)
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

  # Recursive walk: split by `+` (and `-`), evaluating deepRV() / deepRV_st()
  # calls. Everything else is retained in the residual RHS.
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
      if (is.name(head) && identical(as.character(head), "deepRV_st")) {
        call_eval <- e
        call_eval[[1L]] <- quote(brms.deeprv::deepRV_st)
        spec <- eval(call_eval, envir = eval_env)
        if (!inherits(spec, "deepRV_st_call")) {
          stop("`deepRV_st(...)` did not produce a deepRV_st_call object",
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
    stop("formula has no deepRV() or deepRV_st() term. Use brms::brm() ",
         "directly for non-deepRV models.", call. = FALSE)
  }
  if (length(collected) > 1L) {
    stop(sprintf("multiple deepRV / deepRV_st terms in formula (%d); ",
                 length(collected)),
         "v0.1 supports exactly one. Stacking multiple decoders is a ",
         "planned v0.2 feature.", call. = FALSE)
  }
  list(deepRV = collected, rhs = residual, lhs = formula[[2L]])
}
