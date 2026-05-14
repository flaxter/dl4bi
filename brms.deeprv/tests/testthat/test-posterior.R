skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

make_tiny_fit <- (function() {
  cached <- NULL
  function() {
    if (!is.null(cached)) return(cached)
    skip_if_no_rstan()
    skip_on_cran()
    dr <- load_deeprv("unit_interval", grid_size = 10,
                      kernel = "matern_1_2")
    set.seed(99L)
    L <- as.integer(dr$L)
    z_true <- rnorm(L)
    ls_true <- 0.2
    W1 <- dr$weights$W1; b1 <- dr$weights$b1
    W2 <- dr$weights$W2; b2 <- dr$weights$b2
    h <- pmax(crossprod(W1, c(z_true, ls_true)) + b1, 0)
    mu <- as.numeric(crossprod(W2, h) + b2)
    y <- rpois(L, exp(mu))
    df <- data.frame(s = dr$grid_coords, y = y, obs_idx = seq_len(L))
    cached <<- deeprv_brm(
      y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                 ls_prior = prior_uniform(0.05, 0.5)),
      data = df, family = poisson(),
      chains = 2L, iter = 400L, warmup = 200L, seed = 1L, cores = 2L
    )
    cached
  }
})()

test_that("posterior_eta_draws() forward matches Stan's transformed parameter mu", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  # Stan kept mu in transformed parameters; the R-side forward pass
  # should reproduce it draw-by-draw to within float tolerance.
  stan_mu <- rstan::extract(fit$stanfit, pars = "mu")$mu  # (S, L)

  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls"))
  r_mu <- brms.deeprv:::forward_decode_batched(
    fit$decoder, draws$z, draws$ls
  )
  expect_equal(dim(r_mu), dim(stan_mu))
  expect_lt(max(abs(r_mu - stan_mu)), 1e-4)
})

test_that("posterior_eta_draws() recovers eta as fixed + spatial", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  eta <- posterior_eta_draws(fit)
  expect_equal(nrow(eta), 400L)         # 2 chains x 200 post-warmup
  expect_equal(ncol(eta), fit$standata$N)

  # Decompose: eta should equal X * b + mu[obs_idx] for each draw.
  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls", "b"))
  b <- as.matrix(draws$b)
  mu_full <- brms.deeprv:::forward_decode_batched(
    fit$decoder, draws$z, draws$ls
  )
  expected <- b %*% t(fit$X) +
    mu_full[, fit$standata$obs_idx, drop = FALSE]
  expect_lt(max(abs(eta - expected)), 1e-10)
})

test_that("posterior_epred() returns response-scale expected draws", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  epred <- posterior_epred(fit)
  expect_equal(dim(epred), c(400L, fit$standata$N))
  # All values strictly positive for Poisson.
  expect_true(all(epred > 0))
  # Mean of epred draws should be a rough estimate of the rate.
  expect_true(all(is.finite(colMeans(epred))))
})

test_that("posterior_predict() returns integer Poisson draws", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  set.seed(7L)
  ypred <- posterior_predict(fit)
  expect_equal(dim(ypred), c(400L, fit$standata$N))
  expect_true(all(ypred >= 0))
  expect_true(all(ypred == round(ypred)))
})

test_that("posterior_*() refuses newdata in v0.1", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  newdf <- data.frame(s = 0.5)
  expect_error(posterior_predict(fit, newdata = newdf),
               "training grid points")
  expect_error(posterior_epred(fit, newdata = newdf),
               "training grid points")
})
