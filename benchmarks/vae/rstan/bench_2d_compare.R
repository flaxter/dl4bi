# 2D head-to-head: brms exact / brms HSGP / deeprv Kron, on a unit-square
# Poisson fixture drawn from a separable Matern-3/2 GP.
#
#   N_side = 20  (L = 400 grid points)
#   ls_x = ls_y = 0.2
#   sigma_gp = 1
#   intercept = -1 (mean rate ~ 0.37)
#
# All three methods at iter = 4000 / warmup = 2000 / 2 chains, default
# priors. For brms gp() with two inputs we set `iso = FALSE` so each
# axis has its own length-scale, matching the deeprv Kron parameterisation
# (separate ls_x, ls_y).
#
# Per-method we extract S = 4000 posterior draws of eta, then compute
# per-draw RMSE(eta_s, eta_true). Report mean + 5/95th percentiles.

suppressPackageStartupMessages({
  library(rstan)
  library(brms)
  library(brms.deeprv)
})
options(mc.cores = 2L)
rstan_options(auto_write = TRUE)

N_side <- 20L
L      <- N_side * N_side          # = 400
ls_t   <- 0.20
beta0  <- -1
iter   <- 4000L
warmup <- 2000L
seed   <- 1L

# ---- Fixture (separable Matern-3/2 sample via Kronecker Cholesky) ---------
matern32 <- function(s1, s2, l) {
  d <- abs(outer(s1, s2, "-"))
  r <- sqrt(3) * d / l
  (1 + r) * exp(-r)
}
set.seed(seed)
axis_s <- seq(0, 1, length.out = N_side)
K_axis <- matern32(axis_s, axis_s, ls_t) + 1e-6 * diag(N_side)
Lc <- t(chol(K_axis))
Z <- matrix(rnorm(N_side * N_side), nrow = N_side, ncol = N_side)
# Separable Kronecker draw: F = L_x %*% Z %*% L_y^T (here L_x = L_y = Lc).
F_grid <- Lc %*% Z %*% t(Lc)        # (N_side, N_side)
# Flatten column-major to match load_deeprv_kron's grid order
# (flat_idx = (j - 1) * N_side + i  corresponds to F_grid[i, j]).
mu_true <- as.numeric(F_grid)
eta_true <- beta0 + mu_true
y <- rpois(L, exp(eta_true))

df <- data.frame(
  x1      = rep(axis_s, times = N_side),  # x varies fastest = column-major
  x2      = rep(axis_s, each  = N_side),
  y       = y,
  obs_idx = seq_len(L)
)
cat(sprintf("[fixture] N_side=%d  L=%d  ls=%g  beta0=%g  y range=[%d, %d]  non-zero %d/%d\n",
            N_side, L, ls_t, beta0, min(y), max(y), sum(y > 0), L))

# ---- helpers --------------------------------------------------------------
ndiv <- function(stanfit) {
  sp <- rstan::get_sampler_params(stanfit, inc_warmup = FALSE)
  sum(sapply(sp, function(x) sum(x[, "divergent__"])))
}
ntreedepth <- function(stanfit, max_td = 10L) {
  sp <- rstan::get_sampler_params(stanfit, inc_warmup = FALSE)
  sum(sapply(sp, function(x) sum(x[, "treedepth__"] >= max_td)))
}
rhat_max_all <- function(stanfit) {
  suppressWarnings(s <- rstan::summary(stanfit)$summary)
  max(s[, "Rhat"], na.rm = TRUE)
}
per_draw_rmse_summary <- function(eta_draws, truth) {
  truth_mat <- matrix(truth, nrow = nrow(eta_draws),
                      ncol = ncol(eta_draws), byrow = TRUE)
  rmse_s <- sqrt(rowMeans((eta_draws - truth_mat)^2))
  list(mean = mean(rmse_s),
       lo90 = unname(quantile(rmse_s, 0.05)),
       hi90 = unname(quantile(rmse_s, 0.95)),
       posterior_mean_rmse = sqrt(mean((colMeans(eta_draws) - truth)^2)))
}

# ---- 1. brms::gp() exact (separable Matern-3/2) --------------------------
cat("\n=== 1. brms::gp() exact 2D (cov = matern32, iso = FALSE) ===\n")
t0 <- Sys.time()
fit_exact <- brms::brm(
  y ~ gp(x1, x2, cov = "matern32", scale = FALSE, iso = FALSE),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  refresh = 0, control = list(adapt_delta = 0.95),
  backend = "rstan"
)
wall_exact <- as.numeric(Sys.time() - t0, units = "secs")
eta_exact  <- brms::posterior_linpred(fit_exact)
res_exact  <- per_draw_rmse_summary(eta_exact, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_exact, rhat_max_all(fit_exact$fit),
            ndiv(fit_exact$fit), ntreedepth(fit_exact$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_exact$mean, res_exact$lo90, res_exact$hi90,
            res_exact$posterior_mean_rmse))

# ---- 2. brms::gp() HSGP 2D (k = 20 per dim, c = 1.5) ----------------------
cat("\n=== 2. brms::gp() HSGP 2D (k = 20 per dim, c = 1.5) ===\n")
t0 <- Sys.time()
fit_hsgp <- brms::brm(
  formula = stats::as.formula(
    "y ~ gp(x1, x2, cov = 'matern32', scale = FALSE, iso = FALSE, k = 20, c = 1.5)"),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  refresh = 0, control = list(adapt_delta = 0.95),
  backend = "rstan"
)
wall_hsgp <- as.numeric(Sys.time() - t0, units = "secs")
eta_hsgp  <- brms::posterior_linpred(fit_hsgp)
res_hsgp  <- per_draw_rmse_summary(eta_hsgp, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_hsgp, rhat_max_all(fit_hsgp$fit),
            ndiv(fit_hsgp$fit), ntreedepth(fit_hsgp$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_hsgp$mean, res_hsgp$lo90, res_hsgp$hi90,
            res_hsgp$posterior_mean_rmse))

# ---- 3. deeprv_brm() Kron (separable axis-wise Matern-3/2 decoder) -------
cat("\n=== 3. deeprv_brm() Kron (load_deeprv_kron unit_square, grid_side = 20) ===\n")
dr_kron <- load_deeprv_kron("unit_square", grid_side = N_side,
                            kernel = "matern_3_2")
t0 <- Sys.time()
fit_drv <- deeprv_brm(
  y ~ deepRV(x1, decoder = dr_kron, obs_idx = obs_idx,
             ls_prior = prior_uniform(0.05, 0.5)),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  cores = 2L, control = list(adapt_delta = 0.95)
)
wall_drv <- as.numeric(Sys.time() - t0, units = "secs")
eta_drv  <- posterior_eta_draws(fit_drv)
res_drv  <- per_draw_rmse_summary(eta_drv, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_drv, rhat_max_all(fit_drv$stanfit),
            ndiv(fit_drv$stanfit), ntreedepth(fit_drv$stanfit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_drv$mean, res_drv$lo90, res_drv$hi90,
            res_drv$posterior_mean_rmse))

# ---- Summary --------------------------------------------------------------
cat("\n=========== 2D SUMMARY (N_side = 20, L = 400, matern_3_2 separable, iter = 4000) ===========\n")
out <- data.frame(
  method = c("brms exact 2D (iso=F)", "brms HSGP 2D (k=20)", "deeprv_brm Kron"),
  wall_s = c(wall_exact, wall_hsgp, wall_drv),
  rhat   = c(rhat_max_all(fit_exact$fit),
             rhat_max_all(fit_hsgp$fit),
             rhat_max_all(fit_drv$stanfit)),
  divs   = c(ndiv(fit_exact$fit),
             ndiv(fit_hsgp$fit),
             ndiv(fit_drv$stanfit)),
  treedepth = c(ntreedepth(fit_exact$fit),
                ntreedepth(fit_hsgp$fit),
                ntreedepth(fit_drv$stanfit)),
  per_draw_rmse_mean = c(res_exact$mean, res_hsgp$mean, res_drv$mean),
  per_draw_rmse_lo90 = c(res_exact$lo90, res_hsgp$lo90, res_drv$lo90),
  per_draw_rmse_hi90 = c(res_exact$hi90, res_hsgp$hi90, res_drv$hi90),
  posterior_mean_rmse = c(res_exact$posterior_mean_rmse,
                          res_hsgp$posterior_mean_rmse,
                          res_drv$posterior_mean_rmse)
)
print(out, row.names = FALSE, digits = 3)

saveRDS(list(out = out, eta_exact = eta_exact, eta_hsgp = eta_hsgp,
             eta_drv = eta_drv, eta_true = eta_true, y = y, df = df),
        "/tmp/L400_2d_compare.rds")
cat("\nSaved /tmp/L400_2d_compare.rds\n")
