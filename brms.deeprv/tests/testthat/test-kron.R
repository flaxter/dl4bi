skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

# Synthetic 2D Poisson fixture on a 10x10 grid via the same Kronecker
# structure the Stan model evaluates.
make_kron_fixture <- function(seed = 23L) {
  dr <- load_deeprv_kron("unit_square", grid_side = 10,
                          kernel = "matern_1_2")
  axis <- dr$x_decoder
  N <- as.integer(dr$grid_side)
  W1 <- axis$weights$W1; b1 <- axis$weights$b1
  W2 <- axis$weights$W2; b2 <- axis$weights$b2
  decode_r <- function(z, ls) {
    h <- pmax(crossprod(W1, c(z, ls)) + b1, 0)
    as.numeric(crossprod(W2, h) + b2)
  }
  set.seed(seed)
  z <- matrix(rnorm(N * N), nrow = N, ncol = N)
  ls_x <- 0.2; ls_y <- 0.3
  mid <- apply(z, 2L, decode_r, ls = ls_x)              # (N, N)
  mu_grid <- t(apply(mid, 1L, decode_r, ls = ls_y))     # (N, N)
  # Flatten column-major so flat_idx = (j - 1) * N + i corresponds to
  # mu_grid[i, j] - matches the load_deeprv_kron grid_coords ordering.
  mu <- as.numeric(mu_grid)
  y <- rpois(N * N, exp(mu))
  df <- data.frame(
    s_x      = dr$grid_coords[, 1L],
    s_y      = dr$grid_coords[, 2L],
    y        = y,
    obs_idx  = seq_len(N * N)
  )
  list(dr = dr, df = df, mu_grid = mu_grid, ls_x = ls_x, ls_y = ls_y)
}

test_that("deeprv_brm() accepts a deepRV_decoder_kron", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_kron_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s_x, decoder = fix$dr, obs_idx = obs_idx,
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
  expect_true(inherits(fit$decoder, "deepRV_decoder_kron"))
  expect_identical(fit$standata$N_side, 10L)
  expect_identical(fit$standata$L, 100L)
  # Stan should expose both ls_x and ls_y.
  expect_true(all(c("ls_x", "ls_y") %in% fit$stanfit@model_pars))
})

test_that("posterior_eta_draws matches Stan's mu in the Kron model", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_kron_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s_x, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )

  # Compare R-side Kron forward to Stan's `mu`.
  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls_x", "ls_y", "mu"))
  r_mu <- brms.deeprv:::forward_decode_kron_batched(
    fix$dr, draws$z, draws$ls_x, draws$ls_y
  )
  expect_equal(dim(r_mu), dim(draws$mu))
  expect_lt(max(abs(r_mu - draws$mu)), 1e-4)

  # And eta = X * b + mu[obs_idx] picks the right cells.
  eta <- posterior_eta_draws(fit)
  b <- as.matrix(rstan::extract(fit$stanfit, pars = "b")$b)
  expected <- b %*% t(fit$X) +
    draws$mu[, fit$standata$obs_idx, drop = FALSE]
  expect_lt(max(abs(eta - expected)), 1e-10)
})

test_that("conditional_effects() returns a 2D summary for Kron decoders", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_kron_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s_x, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )
  ce <- conditional_effects(fit)
  expect_s3_class(ce, "deeprv_conditional_effects")
  df <- ce$deepRV
  expect_setequal(names(df),
                  c("grid_index", "s_x", "s_y", "estimate", "lower", "upper"))
  expect_equal(nrow(df), 100L)

  pdf(file = NULL); on.exit(dev.off(), add = TRUE)
  expect_invisible(plot(ce))
})

test_that("deeprv_brm rejects by= with Kron decoders", {
  fix <- make_kron_fixture()
  fix$df$x <- runif(nrow(fix$df))
  expect_error(
    deeprv_brm(
      y ~ deepRV(s_x, by = x, decoder = fix$dr, obs_idx = obs_idx,
                 ls_prior = prior_uniform(0.05, 0.5)),
      data = fix$df, family = poisson()),
    "not supported with Kronecker"
  )
})
