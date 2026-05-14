#' Posterior summaries of a deepRV spatial field
#'
#' Returns the posterior mean and credible-interval bounds of the
#' decoder output `mu` evaluated at every training grid point. Mirrors
#' `brms::conditional_effects()` for `brmsfit` objects: a data frame
#' that downstream code (or the provided `plot()` method) can render
#' as the spatial smooth.
#'
#' This is the section 2.7.2 deliverable from `DESIGN.md`. The 2D heatmap
#' for Kronecker fits is deferred to the v0.2 follow-up when
#' `load_deeprv_kron()` becomes a usable fit target.
#'
#' @param x A `deeprv_fit` from [deeprv_brm()].
#' @param probs Numeric pair giving the credible interval (default 90 percent).
#' @param ... Reserved for future arguments.
#'
#' @return A `deeprv_conditional_effects` object: a list of data frames,
#'   one per deepRV term (currently always one). Each data frame has
#'   columns `grid_index`, `s` (the coordinate from `decoder$grid_coords`),
#'   `estimate` (posterior mean), `lower`, `upper`.
#'
#' @export
conditional_effects <- function(x, probs = c(0.05, 0.95), ...) {
  UseMethod("conditional_effects")
}

#' @rdname conditional_effects
#' @export
conditional_effects.deeprv_fit <- function(x, probs = c(0.05, 0.95), ...) {
  if (!is.numeric(probs) || length(probs) != 2L ||
      probs[1] >= probs[2] || probs[1] < 0 || probs[2] > 1) {
    stop("`probs` must be a numeric vector of length 2 with ",
         "0 <= probs[1] < probs[2] <= 1", call. = FALSE)
  }
  grid <- x$decoder$grid_coords
  if (is.matrix(grid)) {
    stop("conditional_effects() for unit_square / Kronecker decoders is ",
         "not implemented in v0.1.", call. = FALSE)
  }
  by_mode <- if (is.null(x$by_mode)) "none" else x$by_mode

  draws <- rstan::extract(x$stanfit, pars = c("z", "ls"))
  ls <- draws$ls

  if (by_mode == "factor") {
    z <- draws$z                                       # (S, G, L)
    G <- dim(z)[2L]
    df_list <- lapply(seq_len(G), function(g) {
      mu_g <- forward_decode_batched(x$decoder, z[, g, ], ls)
      data.frame(
        group      = if (is.null(x$by_levels)) as.character(g) else x$by_levels[g],
        grid_index = seq_along(grid),
        s          = as.numeric(grid),
        estimate   = colMeans(mu_g),
        lower      = apply(mu_g, 2L, stats::quantile, probs = probs[1]),
        upper      = apply(mu_g, 2L, stats::quantile, probs = probs[2])
      )
    })
    df <- do.call(rbind, df_list)
    rownames(df) <- NULL
  } else {
    mu_full <- forward_decode_batched(x$decoder, draws$z, ls)
    df <- data.frame(
      grid_index = seq_along(grid),
      s          = as.numeric(grid),
      estimate   = colMeans(mu_full),
      lower      = apply(mu_full, 2L, stats::quantile, probs = probs[1]),
      upper      = apply(mu_full, 2L, stats::quantile, probs = probs[2])
    )
  }

  out <- list(deepRV = df)
  attr(out, "probs") <- probs
  attr(out, "decoder_label") <- decoder_label(x$decoder)
  attr(out, "by_mode") <- by_mode
  class(out) <- c("deeprv_conditional_effects", "list")
  out
}

#' @export
print.deeprv_conditional_effects <- function(x, ...) {
  probs <- attr(x, "probs")
  cat(sprintf("<deeprv_conditional_effects> (%d%%/%d%% CI)\n",
              round(probs[1] * 100), round(probs[2] * 100)))
  for (nm in names(x)) {
    cat(sprintf("  $%s: %d grid points\n", nm, nrow(x[[nm]])))
  }
  invisible(x)
}

#' Plot a deepRV conditional-effects summary
#'
#' Uses base graphics so the package doesn't pull ggplot2 as a hard
#' dependency. Returns the input invisibly so it composes with
#' `|>` pipelines.
#'
#' @param x A `deeprv_conditional_effects` from [conditional_effects()].
#' @param ... Forwarded to `plot()`.
#' @export
plot.deeprv_conditional_effects <- function(x, ...) {
  df <- x[[1L]]
  decoder_label <- attr(x, "decoder_label")
  by_mode <- attr(x, "by_mode")
  ylim <- range(df$lower, df$upper)
  if (identical(by_mode, "factor") && "group" %in% names(df)) {
    # One panel per group on a shared y-axis; recyclable color palette.
    groups <- unique(df$group)
    G <- length(groups)
    nrow_p <- ceiling(sqrt(G))
    ncol_p <- ceiling(G / nrow_p)
    op <- graphics::par(mfrow = c(nrow_p, ncol_p),
                        oma = c(0, 0, 2, 0))
    on.exit(graphics::par(op), add = TRUE)
    for (g in groups) {
      sub <- df[df$group == g, , drop = FALSE]
      graphics::plot(sub$s, sub$estimate, type = "n",
                     xlab = "s", ylab = "mu(s)",
                     ylim = ylim, main = paste0("group: ", g), ...)
      graphics::polygon(c(sub$s, rev(sub$s)),
                        c(sub$lower, rev(sub$upper)),
                        col = grDevices::adjustcolor("steelblue",
                                                     alpha.f = 0.25),
                        border = NA)
      graphics::lines(sub$s, sub$estimate, lwd = 2, col = "steelblue")
    }
    graphics::mtext(sprintf("deepRV smooth (%s)", decoder_label),
                    outer = TRUE)
  } else {
    graphics::plot(df$s, df$estimate, type = "l",
                   xlab = "s", ylab = "mu(s)", ylim = ylim,
                   main = sprintf("deepRV smooth (%s)", decoder_label),
                   ...)
    graphics::polygon(c(df$s, rev(df$s)),
                      c(df$lower, rev(df$upper)),
                      col = grDevices::adjustcolor("steelblue",
                                                   alpha.f = 0.25),
                      border = NA)
    graphics::lines(df$s, df$estimate, lwd = 2, col = "steelblue")
  }
  invisible(x)
}
