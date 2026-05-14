# End-to-end fit tests. The smoke decoders ship at 1000 training steps,
# so they're too crude for a recovery claim - but they suffice to
# verify the wiring: HMC runs, the parsed formula maps cleanly to Stan
# data, and the sampler finishes without errors. Full-catalog recovery
# tests (DESIGN.md section 2.10) land once the v0.1 catalog is trained.

skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

# Common synthetic Poisson fixture: generate y by sampling decode(z, ls)
# at known z, ls, then taking Poisson draws.
make_poisson_fixture <- function(seed = 42L) {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(seed)
  L <- as.integer(dr$L)
  z_true <- rnorm(L)
  ls_true <- 0.2
  beta0_true <- 0.0
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  relu <- function(x) pmax(x, 0)
  x <- c(z_true, ls_true)
  h <- relu(crossprod(W1, x) + b1)
  mu <- as.numeric(crossprod(W2, h) + b2)
  y <- rpois(L, exp(beta0_true + mu))
  list(dr = dr, df = data.frame(s = dr$grid_coords, y = y,
                                obs_idx = seq_len(L)),
       mu_true = mu, ls_true = ls_true)
}

test_that("deeprv_brm() runs end-to-end on a Poisson fixture", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_poisson_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L
  )

  expect_s3_class(fit, "deeprv_fit")
  expect_true(inherits(fit$stanfit, "stanfit"))
  expect_identical(fit$standata$N, fix$dr$L)
  expect_identical(fit$standata$K, 1L)  # intercept only

  sp <- rstan::get_sampler_params(fit$stanfit, inc_warmup = FALSE)
  ndiv <- sum(sapply(sp, function(s) sum(s[, "divergent__"])))
  expect_lt(ndiv, 10L)

  summary <- rstan::summary(fit$stanfit, pars = c("b", "ls"))$summary
  expect_true(all(is.finite(summary[, "Rhat"])))
  expect_true(all(is.finite(summary[, "n_eff"])))
  expect_true(all(summary[, "n_eff"] > 10))
})

test_that("deeprv_brm() handles RHS covariates via model.matrix", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_poisson_fixture()
  set.seed(7L)
  # Inject a covariate. Truth: y ~ Poisson(exp(mu + 0.3 * x)).
  x_vec <- rnorm(nrow(fix$df))
  beta_x_true <- 0.3
  fix$df$x <- x_vec
  fix$df$y <- rpois(length(fix$mu_true),
                    exp(fix$mu_true + beta_x_true * x_vec))

  fit <- deeprv_brm(
    y ~ x + deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
                   ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L
  )

  # K = 2: intercept + x.
  expect_identical(fit$standata$K, 2L)
  expect_identical(ncol(fit$X), 2L)
  expect_true("(Intercept)" %in% colnames(fit$X))
  expect_true("x" %in% colnames(fit$X))

  draws <- rstan::extract(fit$stanfit, pars = "b")$b
  # First column is the intercept, second is x by model.matrix ordering.
  expect_equal(dim(draws)[2L], 2L)
  # Posterior mean of beta_x should be in the ballpark of 0.3 (loose -
  # smoke decoder, L=10).
  expect_lt(abs(mean(draws[, 2L]) - beta_x_true), 1.0)
})

test_that("deeprv_brm() drops the intercept when RHS specifies -1", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_poisson_fixture()
  set.seed(11L)
  x_vec <- rnorm(nrow(fix$df))
  fix$df$x <- x_vec

  fit <- deeprv_brm(
    y ~ x - 1 + deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
                        ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 1L,
    iter    = 200L,
    warmup  = 100L,
    seed    = 1L,
    cores   = 1L
  )
  expect_identical(fit$standata$K, 1L)
  expect_identical(colnames(fit$X), "x")
})

test_that("deeprv_brm() supports gaussian() family", {
  skip_if_no_rstan()
  skip_on_cran()

  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(13L)
  L <- as.integer(dr$L)
  z_true <- rnorm(L)
  ls_true <- 0.2
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  h <- pmax(crossprod(W1, c(z_true, ls_true)) + b1, 0)
  mu <- as.numeric(crossprod(W2, h) + b2)
  sigma_true <- 0.3
  y <- mu + rnorm(L, sd = sigma_true)
  df <- data.frame(s = dr$grid_coords, y = y, obs_idx = seq_len(L))

  fit <- deeprv_brm(
    y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = df,
    family  = gaussian(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L
  )
  expect_identical(fit$family$family, "gaussian")
  # sigma should appear in the stanfit parameter list.
  expect_true("sigma" %in% fit$stanfit@model_pars)
  sigma_draws <- rstan::extract(fit$stanfit, pars = "sigma")$sigma
  expect_true(all(sigma_draws > 0))
})

test_that("deeprv_brm() rejects v0.2 knobs and unsupported families", {
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
               data = df, family = binomial()),
    "supported families"
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
