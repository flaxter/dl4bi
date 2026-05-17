# Where HSGP breaks: fix K = 20 but make the truth's length-scale
# short enough that K is too small to represent the spectrum.
#
# Rule of thumb (Riutort-Mayol et al. 2023): K >= 1.75 * c * L_dom / ls
# For ls = 0.05, c = 1.5, L_dom = 1 -> K >= 52.5. So K = 20 is way under
# and HSGP should oversmooth visibly.
#
# Four methods at iter = 4000, default priors:
#   1. brms exact         -- the reference
#   2. brms HSGP k = 20   -- the headline default; should oversmooth here
#   3. brms HSGP k = 60   -- control; sufficient K for ls = 0.05
#   4. deeprv_brm         -- L=250 matern_3_2 decoder (already trained)

suppressPackageStartupMessages({
  library(rstan); library(brms); library(brms.deeprv)
})
options(brms.deeprv.decoder_dir = "/tmp/dl4bi_L250")
options(mc.cores = 2L)
rstan_options(auto_write = TRUE)

L      <- 250L
ls_t   <- 0.05       # short -- below K=20 HSGP's resolution
beta0  <- 0          # higher signal-to-noise than the L=250 fixture (mean rate 1)
iter   <- 4000L
warmup <- 2000L
seed   <- 1L

# ---- Fixture --------------------------------------------------------------
set.seed(seed)
dr <- load_deeprv("unit_interval", grid_size = L, kernel = "matern_3_2")
s  <- as.numeric(dr$grid_coords)
matern32 <- function(s1, s2, l) {
  d <- abs(outer(s1, s2, "-"))
  r <- sqrt(3) * d / l
  (1 + r) * exp(-r)
}
K <- matern32(s, s, ls_t) + 1e-6 * diag(L)
Lc <- t(chol(K))
mu_true  <- as.numeric(Lc %*% rnorm(L))
eta_true <- beta0 + mu_true
y <- rpois(L, exp(eta_true))
df <- data.frame(s = s, y = y, obs_idx = seq_along(s))
cat(sprintf("[fixture] L=%d ls_TRUE=%g beta0=%g  y range=[%d, %d]  non-zero %d/%d\n",
            L, ls_t, beta0, min(y), max(y), sum(y > 0), L))
cat(sprintf("[HSGP rule-of-thumb] K_required >= 1.75 * 1.5 / %g = %.1f\n",
            ls_t, 1.75 * 1.5 / ls_t))

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
fit_hsgp <- function(k_val, c_val = 1.5) {
  formula_str <- sprintf(
    "y ~ gp(s, cov = 'matern32', scale = FALSE, k = %d, c = %g)",
    as.integer(k_val), c_val)
  t0 <- Sys.time()
  fit <- brms::brm(
    formula = stats::as.formula(formula_str),
    data = df, family = poisson(),
    chains = 2L, iter = iter, warmup = warmup, seed = seed,
    refresh = 0, control = list(adapt_delta = 0.95),
    backend = "rstan"
  )
  wall <- as.numeric(Sys.time() - t0, units = "secs")
  list(fit = fit, wall = wall)
}

# ---- 1. brms::gp() exact --------------------------------------------------
cat("\n=== 1. brms::gp() exact (cov = matern32, default priors) ===\n")
t0 <- Sys.time()
fit_exact <- brms::brm(
  y ~ gp(s, cov = "matern32", scale = FALSE),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  refresh = 0, control = list(adapt_delta = 0.95), backend = "rstan"
)
wall_exact <- as.numeric(Sys.time() - t0, units = "secs")
eta_exact <- brms::posterior_linpred(fit_exact)
res_exact <- per_draw_rmse_summary(eta_exact, eta_true)
cat(sprintf("  wall = %.0fs  Rhat = %.3f  divs = %d  tree>= = %d\n",
            wall_exact, rhat_max_all(fit_exact$fit),
            ndiv(fit_exact$fit), ntreedepth(fit_exact$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_exact$mean, res_exact$lo90, res_exact$hi90,
            res_exact$posterior_mean_rmse))

# ---- 2. brms HSGP k = 20 (should oversmooth) -----------------------------
cat("\n=== 2. brms HSGP (k = 20, c = 1.5) - INSUFFICIENT K for ls_t = 0.05 ===\n")
r <- fit_hsgp(20)
eta <- brms::posterior_linpred(r$fit)
res_hsgp20 <- per_draw_rmse_summary(eta, eta_true)
cat(sprintf("  wall = %.0fs  Rhat = %.3f  divs = %d  tree>= = %d\n",
            r$wall, rhat_max_all(r$fit$fit),
            ndiv(r$fit$fit), ntreedepth(r$fit$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_hsgp20$mean, res_hsgp20$lo90, res_hsgp20$hi90,
            res_hsgp20$posterior_mean_rmse))
wall_hsgp20 <- r$wall; eta_hsgp20 <- eta; fit_hsgp20 <- r$fit

# ---- 3. brms HSGP k = 60 (control, sufficient K) -------------------------
cat("\n=== 3. brms HSGP (k = 60, c = 1.5) - SUFFICIENT K control ===\n")
r <- fit_hsgp(60)
eta <- brms::posterior_linpred(r$fit)
res_hsgp60 <- per_draw_rmse_summary(eta, eta_true)
cat(sprintf("  wall = %.0fs  Rhat = %.3f  divs = %d  tree>= = %d\n",
            r$wall, rhat_max_all(r$fit$fit),
            ndiv(r$fit$fit), ntreedepth(r$fit$fit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_hsgp60$mean, res_hsgp60$lo90, res_hsgp60$hi90,
            res_hsgp60$posterior_mean_rmse))
wall_hsgp60 <- r$wall; eta_hsgp60 <- eta; fit_hsgp60 <- r$fit

# ---- 4. deeprv_brm() ------------------------------------------------------
cat("\n=== 4. deeprv_brm() (matern_3_2 L=250 decoder, prior_uniform(0.02, 0.5)) ===\n")
t0 <- Sys.time()
fit_drv <- deeprv_brm(
  y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
             ls_prior = prior_uniform(0.02, 0.5)),
  data = df, family = poisson(),
  chains = 2L, iter = iter, warmup = warmup, seed = seed,
  cores = 2L, control = list(adapt_delta = 0.95)
)
wall_drv <- as.numeric(Sys.time() - t0, units = "secs")
eta_drv  <- posterior_eta_draws(fit_drv)
res_drv  <- per_draw_rmse_summary(eta_drv, eta_true)
cat(sprintf("  wall = %.0fs  Rhat = %.3f  divs = %d  tree>= = %d\n",
            wall_drv, rhat_max_all(fit_drv$stanfit),
            ndiv(fit_drv$stanfit), ntreedepth(fit_drv$stanfit)))
cat(sprintf("  per-draw RMSE: mean = %.3f  90%% CI [%.3f, %.3f]   posterior-mean RMSE = %.3f\n",
            res_drv$mean, res_drv$lo90, res_drv$hi90,
            res_drv$posterior_mean_rmse))

# ---- Summary --------------------------------------------------------------
cat(sprintf("\n=========== SUMMARY (L = 250, matern_3_2, ls_TRUE = %g, iter = 4000) ===========\n", ls_t))
out <- data.frame(
  method = c("brms exact", "brms HSGP k=20", "brms HSGP k=60", "deeprv_brm"),
  wall_s = c(wall_exact, wall_hsgp20, wall_hsgp60, wall_drv),
  rhat   = c(rhat_max_all(fit_exact$fit),
             rhat_max_all(fit_hsgp20$fit),
             rhat_max_all(fit_hsgp60$fit),
             rhat_max_all(fit_drv$stanfit)),
  divs   = c(ndiv(fit_exact$fit), ndiv(fit_hsgp20$fit),
             ndiv(fit_hsgp60$fit), ndiv(fit_drv$stanfit)),
  per_draw_rmse_mean = c(res_exact$mean, res_hsgp20$mean,
                         res_hsgp60$mean, res_drv$mean),
  per_draw_rmse_lo90 = c(res_exact$lo90, res_hsgp20$lo90,
                         res_hsgp60$lo90, res_drv$lo90),
  per_draw_rmse_hi90 = c(res_exact$hi90, res_hsgp20$hi90,
                         res_hsgp60$hi90, res_drv$hi90),
  posterior_mean_rmse = c(res_exact$posterior_mean_rmse,
                          res_hsgp20$posterior_mean_rmse,
                          res_hsgp60$posterior_mean_rmse,
                          res_drv$posterior_mean_rmse)
)
print(out, row.names = FALSE, digits = 3)

saveRDS(list(out = out, eta_exact = eta_exact, eta_hsgp20 = eta_hsgp20,
             eta_hsgp60 = eta_hsgp60, eta_drv = eta_drv,
             eta_true = eta_true, y = y, s = s),
        "/tmp/short_ls_compare.rds")
cat("\nSaved /tmp/short_ls_compare.rds\n")
