skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

test_that("deeprv_brm() with by = <numeric> fits an SVC model end-to-end", {
  skip_if_no_rstan()
  skip_on_cran()

  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(123L)
  L <- as.integer(dr$L)
  z_true <- rnorm(L)
  ls_true <- 0.2
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  h <- pmax(crossprod(W1, c(z_true, ls_true)) + b1, 0)
  mu <- as.numeric(crossprod(W2, h) + b2)

  # x_i in [0.5, 1.5] so x_i * mu[i] is well-scaled.
  x_vec <- runif(L, 0.5, 1.5)
  y <- rpois(L, exp(x_vec * mu))
  df <- data.frame(s = dr$grid_coords, y = y, obs_idx = seq_len(L),
                   x = x_vec)

  fit <- deeprv_brm(
    y ~ deepRV(s, by = x, decoder = dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = df,
    family  = poisson(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L
  )
  expect_identical(fit$by_mode, "svc")
  expect_equal(fit$by_values, x_vec)
  expect_true("x_by" %in% names(fit$standata))
  expect_equal(fit$standata$x_by, x_vec)
})

test_that("posterior_eta_draws() applies the SVC multiplier", {
  skip_if_no_rstan()
  skip_on_cran()

  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(7L)
  L <- as.integer(dr$L)
  x_vec <- runif(L, 0.5, 1.5)
  y <- rpois(L, 1)
  df <- data.frame(s = dr$grid_coords, y = y, obs_idx = seq_len(L),
                   x = x_vec)

  fit <- deeprv_brm(
    y ~ deepRV(s, by = x, decoder = dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data = df, family = poisson(),
    chains = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )

  eta <- posterior_eta_draws(fit)
  # Recompute expected eta: X * b + x_vec * mu[obs_idx].
  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls", "b"))
  b <- as.matrix(draws$b)
  mu_full <- brms.deeprv:::forward_decode_batched(
    fit$decoder, draws$z, draws$ls
  )
  spatial <- sweep(mu_full[, fit$standata$obs_idx, drop = FALSE],
                   2L, x_vec, FUN = "*")
  expected <- b %*% t(fit$X) + spatial
  expect_lt(max(abs(eta - expected)), 1e-10)
})

test_that("deeprv_brm() catches length mismatch on numeric by", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  df <- data.frame(s = dr$grid_coords, y = rep(1L, dr$L),
                   obs_idx = seq_len(dr$L))
  # Hand in a `by` vector longer than nrow(data) by evaluating in a scope
  # that has the longer vector available.
  bigger <- rnorm(dr$L + 3L)
  expect_error(
    deeprv_brm(y ~ deepRV(s, by = bigger, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5)),
               data = df, family = poisson()),
    "must match nrow"
  )
})
