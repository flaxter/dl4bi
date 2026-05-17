# Apples-to-apples (well, default-priors) Poisson recovery at L = 250
# with a harder fixture than the RBF-at-ls=0.2 baseline:
#
#   kernel = Matern-3/2
#   ls = 0.15  (tighter than 0.2)
#   beta0 = -1 (mean rate ~ 0.37 -> more Poisson noise per cell)
#
# All three methods at iter = 4000, default priors:
#   - brms::gp(s, cov = "matern32", scale = FALSE)
#   - brms::gp(s, cov = "matern32", scale = FALSE, k = 20, c = 1.5)
#   - deeprv_brm with kernel = "matern_3_2"
#
# For each method, extract S posterior draws of eta = X*b + spatial,
# then compute per-draw RMSE(eta_s, truth). Report mean and 90% CI of
# the per-draw distribution.
#
# Run: Rscript benchmarks/vae/rstan/bench_L250_compare.R

suppressPackageStartupMessages({
  library(rstan)
  library(brms)
  library(brms.deeprv)
})
options(brms.deeprv.decoder_dir = "/tmp/dl4bi_L250")
options(mc.cores = 2L)
rstan_options(auto_write = TRUE)

L      <- 250L
ls_t   <- 0.15
beta0  <- -1
iter   <- 4000L
warmup <- 2000L
seed   <- 1L

# ---- Fixture --------------------------------------------------------------
set.seed(seed)
dr <- load_deeprv("unit_interval", grid_size = L, kernel = "matern_3_2")
s  <- as.numeric(dr$grid_coords)

# Matern-3/2 kernel: K(d) = (1 + sqrt(3) d / l) * exp(-sqrt(3) d / l).
matern32 <- function(s1, s2, l) {
  d <- abs(outer(s1, s2, "-"))
  r <- sqrt(3) * d / l
  (1 + r) * exp(-r)
}
K <- matern32(s, s, ls_t) + 1e-6 * diag(L)
Lc <- t(chol(K))
mu_true <- as.numeric(Lc %*% rnorm(L))
eta_true <- beta0 + mu_true
y <- rpois(L, exp(eta_true))
df <- data.frame(s = s, y = y, obs_idx = seq_along(s))
cat(sprintf("[fixture] L=%d ls=%g beta0=%g, y range=[%d, %d], non-zero %d/%d\n",
            L, ls_t, beta0, min(y), max(y), sum(y > 0), L))

# Helper: extract S draws of eta per method, return (S, L) matrix.
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
# Per-draw RMSE summary: mean + 90% CI of (RMSE_s) across S posterior draws.
per_draw_rmse_summary <- function(eta_draws, truth) {
  rmse_s <- sqrt(rowMeans((eta_draws - matrix(truth, nrow = nrow(eta_draws),
                                              ncol = ncol(eta_draws),
                                              byrow = TRUE))^2))
  list(mean = mean(rmse_s),
       lo90 = unname(quantile(rmse_s, 0.05)),
       hi90 = unname(quantile(rmse_s, 0.95)),
       posterior_mean_rmse = sqrt(mean((colMeans(eta_draws) - truth)^2)))
}

# ---- 1. brms::gp() exact --------------------------------------------------
cat("\n=== 1. brms::gp() exact (cov = matern32, default priors) ===\n")
t0 <- Sys.time()
fit_exact <- brms::brm(
  y ~ gp(s, cov = "matern32", scale = FALSE),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  refresh = 0, control = list(adapt_delta = 0.95),
  backend = "rstan"
)
wall_exact <- as.numeric(Sys.time() - t0, units = "secs")
eta_exact <- brms::posterior_linpred(fit_exact)  # (S, N)
res_exact <- per_draw_rmse_summary(eta_exact, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_exact, rhat_max_all(fit_exact$fit),
            ndiv(fit_exact$fit), ntreedepth(fit_exact$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_exact$mean, res_exact$lo90, res_exact$hi90,
            res_exact$posterior_mean_rmse))

# ---- 2. brms::gp() HSGP ---------------------------------------------------
cat("\n=== 2. brms::gp() HSGP (k=20, c=1.5, default priors) ===\n")
t0 <- Sys.time()
fit_hsgp <- brms::brm(
  formula = stats::as.formula(
    "y ~ gp(s, cov = 'matern32', scale = FALSE, k = 20, c = 1.5)"),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  refresh = 0, control = list(adapt_delta = 0.95),
  backend = "rstan"
)
wall_hsgp <- as.numeric(Sys.time() - t0, units = "secs")
eta_hsgp <- brms::posterior_linpred(fit_hsgp)
res_hsgp <- per_draw_rmse_summary(eta_hsgp, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_hsgp, rhat_max_all(fit_hsgp$fit),
            ndiv(fit_hsgp$fit), ntreedepth(fit_hsgp$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_hsgp$mean, res_hsgp$lo90, res_hsgp$hi90,
            res_hsgp$posterior_mean_rmse))

# ---- 3. deeprv_brm() ------------------------------------------------------
cat("\n=== 3. deeprv_brm() (matern_3_2 decoder, prior_uniform(0.05, 0.5)) ===\n")
t0 <- Sys.time()
fit_drv <- deeprv_brm(
  y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
             ls_prior = prior_uniform(0.05, 0.5)),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  cores = 2L, control = list(adapt_delta = 0.95)
)
wall_drv <- as.numeric(Sys.time() - t0, units = "secs")
eta_drv <- posterior_eta_draws(fit_drv)
res_drv <- per_draw_rmse_summary(eta_drv, eta_true)
cat(sprintf("  wall = %.0fs   Rhat = %.3f   divs = %d   tree>= = %d\n",
            wall_drv, rhat_max_all(fit_drv$stanfit),
            ndiv(fit_drv$stanfit), ntreedepth(fit_drv$stanfit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_drv$mean, res_drv$lo90, res_drv$hi90,
            res_drv$posterior_mean_rmse))

# ---- Summary table --------------------------------------------------------
cat("\n=========== SUMMARY (L = 250, matern_3_2, ls_true = 0.15, iter = 4000) ===========\n")
out <- data.frame(
  method = c("brms exact", "brms HSGP (k=20)", "deeprv_brm"),
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
             eta_drv = eta_drv, eta_true = eta_true, y = y, s = s),
        "/tmp/L250_compare.rds")
cat("\nSaved /tmp/L250_compare.rds\n")
