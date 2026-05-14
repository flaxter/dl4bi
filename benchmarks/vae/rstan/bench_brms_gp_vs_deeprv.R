# Wall-clock and posterior comparison: brms::gp() (exact O(L^3) GP) vs
# brms.deeprv::deeprv_brm() (pre-trained MLP decoder) on the same
# Poisson fixture, varying grid size.
#
# Same kernel family on both sides:
#   - brms::gp(s, cov = "exp_quad", scale = FALSE)
#   - deeprv_brm with kernel = "rbf"
#
# Both decoders / kernels match the RBF kernel
#   K(s, s') = exp(-(s - s')^2 / (2 * ls^2))
# with the amplitude held at 1 (the decoder's training assumption);
# brms is configured with sdgp ~ constant(1) to match.
#
# Run: Rscript benchmarks/vae/rstan/bench_brms_gp_vs_deeprv.R
#
# Output: a markdown table printed to stdout (and appended to
# benchmarks/vae/rstan/bench_brms_gp.md when --write).

suppressPackageStartupMessages({
  library(rstan)
  library(brms)
  library(brms.deeprv)
})
options(mc.cores = 2L)
rstan_options(auto_write = TRUE)

# ---- arguments -------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
write_md <- "--write" %in% args
align_priors <- "--align-priors" %in% args
grids <- if (any(grepl("--grids=", args))) {
  spec <- sub("--grids=", "", args[grepl("--grids=", args)])
  as.integer(strsplit(spec, ",", fixed = TRUE)[[1L]])
} else c(20L, 50L)
iter <- if (any(grepl("--iter=", args))) {
  as.integer(sub("--iter=", "", args[grepl("--iter=", args)]))
} else 1000L
warmup <- floor(iter / 2)
# Optional override of the deeprv decoder dir, e.g. for swapping in a
# longer-trained decoder for the same (grid_size, kernel).
if (any(grepl("--decoder-dir=", args))) {
  d <- sub("--decoder-dir=", "", args[grepl("--decoder-dir=", args)])
  options(brms.deeprv.decoder_dir = d)
  cat(sprintf("[deeprv decoder dir overridden to %s]\n", d))
}
ls_lo  <- 0.05
ls_hi  <- 0.5
seed   <- 1L

cat(sprintf("[bench config] grids=%s iter=%d (warmup %d) align_priors=%s\n",
            paste(grids, collapse = ","), iter, warmup, align_priors))

# ---- synthetic fixture per grid -------------------------------------------
make_fixture <- function(L, ls_true = 0.2, beta0_true = 0.0, seed = 1L) {
  set.seed(seed)
  dr <- load_deeprv("unit_interval", grid_size = L, kernel = "rbf")
  s <- as.numeric(dr$grid_coords)
  K <- outer(s, s, function(a, b) exp(-(a - b)^2 / (2 * ls_true^2)))
  K <- K + 1e-6 * diag(L)
  Lc <- t(chol(K))
  mu_true <- as.numeric(Lc %*% rnorm(L))
  y <- rpois(L, exp(beta0_true + mu_true))
  data.frame(s = s, y = y, obs_idx = seq_along(s),
             mu_true = mu_true)
}

# ---- fit via brms::gp() ----------------------------------------------------
# When align_priors = TRUE, attach a uniform(0.05, 0.5) prior to brms's
# `lscale` for the gp(s) term and fix sdgp = 1 via prior(constant(1))
# to match the deeprv side. coef="gps" is the brms-internal name for
# `gp(s, ...)`; we discovered it via brms::default_prior().
fit_brms_gp <- function(df, iter, warmup, seed, align_priors = FALSE) {
  brms_prior <- NULL
  if (align_priors) {
    # brms refuses `coef` + `lb`/`ub` together; the uniform() density
    # is -Inf outside its support, so the posterior is bounded anyway
    # even with the default parameter bounds of <lower=0>.
    # We don't fix sdgp at 1 - brms generates Stan that assigns a
    # vector[Kgp_1] which conflicts with `constant(1)` being scalar int.
    # Instead, with sigma_gp = 1 in the truth and an informative
    # likelihood, brms's default student_t(3, 0, 2.5) prior on sdgp
    # will concentrate the posterior near 1.
    brms_prior <- brms::prior_string(sprintf("uniform(%g, %g)", ls_lo, ls_hi),
                                     class = "lscale", coef = "gps")
  }
  t0 <- Sys.time()
  fit <- brms::brm(
    y ~ gp(s, cov = "exp_quad", scale = FALSE),
    data    = df,
    family  = poisson(),
    prior   = brms_prior,
    chains  = 2L,
    iter    = iter,
    warmup  = warmup,
    seed    = seed,
    refresh = 0,
    control = list(adapt_delta = 0.95),
    backend = "rstan"
  )
  wall <- as.numeric(Sys.time() - t0, units = "secs")
  list(fit = fit, wall = wall)
}

# ---- fit via deeprv_brm() --------------------------------------------------
fit_deeprv <- function(df, ls_lo, ls_hi, iter, warmup, seed) {
  dr <- load_deeprv("unit_interval", grid_size = nrow(df), kernel = "rbf")
  t0 <- Sys.time()
  fit <- deeprv_brm(
    y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(ls_lo, ls_hi)),
    data    = df,
    family  = poisson(),
    chains  = 2L,
    iter    = iter,
    warmup  = warmup,
    seed    = seed,
    cores   = 2L,
    control = list(adapt_delta = 0.95)
  )
  wall <- as.numeric(Sys.time() - t0, units = "secs")
  list(fit = fit, wall = wall)
}

ndiv <- function(stanfit) {
  sp <- rstan::get_sampler_params(stanfit, inc_warmup = FALSE)
  sum(sapply(sp, function(s) sum(s[, "divergent__"])))
}

# Max Rhat across all sampled parameters (not just the named ones, since
# brms's coef names differ from ours).
rhat_max_all <- function(stanfit) {
  suppressWarnings({
    s <- rstan::summary(stanfit)$summary
  })
  max(s[, "Rhat"], na.rm = TRUE)
}

# ---- benchmark loop --------------------------------------------------------
rows <- list()
for (L in grids) {
  cat(sprintf("\n=== L = %d ===\n", L))
  df <- make_fixture(L, seed = seed)

  cat(sprintf("  fitting brms::gp() ...\n"))
  r_brms <- fit_brms_gp(df, iter, warmup, seed, align_priors = align_priors)
  # posterior_linpred returns (S, N) draws of eta on the link scale.
  brms_eta <- brms::posterior_linpred(r_brms$fit)
  brms_post_mean <- colMeans(brms_eta)
  brms_ndiv <- ndiv(r_brms$fit$fit)
  brms_rhat <- rhat_max_all(r_brms$fit$fit)

  cat(sprintf("  fitting deeprv_brm() ...\n"))
  r_drv <- fit_deeprv(df, ls_lo, ls_hi, iter, warmup, seed)
  drv_eta <- posterior_eta_draws(r_drv$fit)
  drv_post_mean <- colMeans(drv_eta)
  drv_ndiv <- ndiv(r_drv$fit$stanfit)
  drv_rhat <- rhat_max_all(r_drv$fit$stanfit)

  # Per-grid posterior mean discrepancy of eta: deeprv vs brms.
  eta_diff <- drv_post_mean - brms_post_mean
  rmse <- sqrt(mean(eta_diff^2))
  truth <- df$mu_true  # truth was generated with beta0 = 0, so eta = mu
  truth_rmse_brms <- sqrt(mean((brms_post_mean - truth)^2))
  truth_rmse_drv  <- sqrt(mean((drv_post_mean - truth)^2))

  rows[[length(rows) + 1L]] <- data.frame(
    L = L,
    brms_wall_s = r_brms$wall, drv_wall_s = r_drv$wall,
    speedup = r_brms$wall / r_drv$wall,
    brms_rhat = brms_rhat, drv_rhat = drv_rhat,
    brms_ndiv = brms_ndiv, drv_ndiv = drv_ndiv,
    eta_rmse_drv_vs_brms = rmse,
    truth_rmse_brms = truth_rmse_brms,
    truth_rmse_drv  = truth_rmse_drv
  )
}

out <- do.call(rbind, rows)
print(out, row.names = FALSE)

# Optional: append a Markdown table to the results file.
if (write_md) {
  md_path <- "benchmarks/vae/rstan/bench_brms_gp.md"
  rendered <- knitr::kable(out, format = "markdown", digits = 3)
  cat(rendered, file = md_path, sep = "\n", append = TRUE)
  cat(sprintf("\nWrote %s\n", md_path))
}
