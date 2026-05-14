skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

make_two_group_fixture <- function(seed = 41L) {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(seed)
  L <- as.integer(dr$L)
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  decode_r <- function(z, ls) {
    h <- pmax(crossprod(W1, c(z, ls)) + b1, 0)
    as.numeric(crossprod(W2, h) + b2)
  }
  mu_a <- decode_r(rnorm(L), 0.15)
  mu_b <- decode_r(rnorm(L), 0.35)
  rows <- expand.grid(s_idx = seq_len(L), group = c("a", "b"),
                      stringsAsFactors = FALSE)
  rows$group <- factor(rows$group, levels = c("a", "b"))
  rows$s <- dr$grid_coords[rows$s_idx]
  rows$obs_idx <- rows$s_idx
  rows$mu_true <- ifelse(rows$group == "a", mu_a[rows$s_idx],
                                            mu_b[rows$s_idx])
  rows$y <- rpois(nrow(rows), exp(rows$mu_true))
  list(dr = dr, df = rows)
}

test_that("ls_pooling = \"none\" fits with vector[G] ls in Stan", {
  skip_if_no_rstan()
  skip_on_cran()
  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5),
               ls_pooling = "none"),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L, iter = 400L, warmup = 200L, seed = 1L, cores = 2L
  )
  expect_identical(fit$ls_pooling, "none")
  ls_dim <- dim(rstan::extract(fit$stanfit, pars = "ls")$ls)
  expect_identical(ls_dim, c(400L, 2L))
})

test_that("ls_pooling = \"partial\" exposes mu_ls and tau_ls", {
  skip_if_no_rstan()
  skip_on_cran()
  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5),
               ls_pooling = "partial"),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L, iter = 400L, warmup = 200L, seed = 1L, cores = 2L
  )
  expect_identical(fit$ls_pooling, "partial")
  expect_true(all(c("ls", "mu_ls", "tau_ls") %in% fit$stanfit@model_pars))
  tau <- rstan::extract(fit$stanfit, pars = "tau_ls")$tau_ls
  expect_true(all(tau > 0))
})

test_that("posterior_eta_draws() indexes per-group ls correctly", {
  skip_if_no_rstan()
  skip_on_cran()
  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5),
               ls_pooling = "none"),
    data = fix$df, family = poisson(),
    chains = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )
  eta <- posterior_eta_draws(fit)
  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls", "b", "mu"))
  b <- as.matrix(draws$b)
  group_idx <- fit$standata$group_idx
  obs_idx <- fit$standata$obs_idx
  N <- length(group_idx)
  expected <- matrix(0, nrow = nrow(draws$mu), ncol = N)
  for (n in seq_len(N)) {
    expected[, n] <- draws$mu[, group_idx[n], obs_idx[n]]
  }
  expected <- b %*% t(fit$X) + expected
  expect_lt(max(abs(eta - expected)), 1e-3)
})

test_that("ls_pooling rejects bad combinations with clear errors", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  df <- data.frame(s = dr$grid_coords, y = rep(1L, dr$L),
                   obs_idx = seq_len(dr$L))
  # No by -> ls_pooling != "complete" is meaningless.
  expect_error(
    deeprv_brm(y ~ deepRV(s, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5),
                          ls_pooling = "none"),
               data = df, family = poisson()),
    "only makes sense"
  )
  # Garbage string.
  df$group <- factor(rep("a", dr$L))
  expect_error(
    deeprv_brm(y ~ deepRV(s, by = group, decoder = dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5),
                          ls_pooling = "garbage"),
               data = df, family = poisson()),
    "one of"
  )
})
