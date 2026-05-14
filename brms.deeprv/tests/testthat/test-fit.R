# End-to-end MVP test. The smoke decoders ship at 1000 training steps,
# so they're too crude for a recovery claim — but they suffice to
# verify the wiring: HMC runs, the parsed formula maps cleanly to Stan
# data, and the sampler finishes without errors. The full-catalog
# recovery test (DESIGN.md section 2.10) lands once the v0.1 catalog is
# trained.

skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

test_that("deeprv_brm() runs end-to-end on a synthetic Poisson fixture", {
  skip_if_no_rstan()
  skip_on_cran()

  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  # Synthesize ground-truth y by sampling beta0 + decode(z, ls) at known
  # values, then taking Poisson draws. The recovery checks here only
  # assert that the wiring works — Rhat finite, ESS positive, no errors.
  set.seed(42L)
  L <- as.integer(dr$L)
  z_true <- rnorm(L)
  ls_true <- 0.2
  beta0_true <- 0.0
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  # Manual forward in R to avoid depending on Stan being compiled here.
  relu <- function(x) pmax(x, 0)
  x <- c(z_true, ls_true)
  h <- relu(crossprod(W1, x) + b1)
  mu <- as.numeric(crossprod(W2, h) + b2)

  y <- rpois(L, exp(beta0_true + mu))
  df <- data.frame(s = dr$grid_coords, y = y, obs_idx = seq_len(L))

  fit <- deeprv_brm(
    y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = df,
    family  = poisson(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L,
    control = list(adapt_delta = 0.95)
  )

  expect_s3_class(fit, "deeprv_fit")
  expect_true(inherits(fit$stanfit, "stanfit"))
  expect_identical(fit$standata$N, L)
  expect_identical(fit$standata$L, L)

  # Sampler should complete with no divergences and finite diagnostics.
  sp <- rstan::get_sampler_params(fit$stanfit, inc_warmup = FALSE)
  ndiv <- sum(sapply(sp, function(s) sum(s[, "divergent__"])))
  expect_lt(ndiv, 10L)  # smoke decoder; loose ceiling

  summary <- rstan::summary(fit$stanfit, pars = c("beta0", "ls"))$summary
  expect_true(all(is.finite(summary[, "Rhat"])))
  expect_true(all(is.finite(summary[, "n_eff"])))
  expect_true(all(summary[, "n_eff"] > 10))  # very loose; just non-trivially informative
})

test_that("deeprv_brm() rejects non-MVP knobs with clear errors", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  df <- data.frame(s = dr$grid_coords, y = rep(1L, dr$L),
                   obs_idx = seq_len(dr$L), region = rep("a", dr$L))

  expect_error(
    deeprv_brm(y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5),
                          by = region),
               data = df, family = poisson()),
    "by"
  )
  expect_error(
    deeprv_brm(y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5),
                          gr = TRUE),
               data = df, family = poisson()),
    "gr"
  )
  expect_error(
    deeprv_brm(y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5)),
               data = df, family = gaussian()),
    "poisson"
  )
})

test_that("deeprv_brm() validates that the prior fits the trained range", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  df <- data.frame(s = dr$grid_coords, y = rep(1L, dr$L),
                   obs_idx = seq_len(dr$L))
  expect_error(
    deeprv_brm(y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 1.5)),
               data = df, family = poisson()),
    "extends outside"
  )
})
