skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

# Synthetic space-time Poisson fixture: L=10 spatial grid, T=5 time
# steps. Per the RW structure: F[t, :] = F[t-1, :] + sigma_t * eps[t, :]
# where eps[t, :] = decode(z[t, :], ls). Roll out manually in R to seed
# the fixture, then ask deeprv_brm() to recover the structure (loosely -
# the smoke decoder isn't well-trained at L=10).
make_st_fixture <- function(seed = 51L) {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(seed)
  L <- as.integer(dr$L)
  T_full <- 5L
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  decode_r <- function(z, ls) {
    h <- pmax(crossprod(W1, c(z, ls)) + b1, 0)
    as.numeric(crossprod(W2, h) + b2)
  }
  ls_true <- 0.2
  sigma_t_true <- 0.3
  z <- matrix(rnorm(T_full * L), nrow = T_full, ncol = L)
  F <- matrix(0, nrow = T_full, ncol = L)
  F[1, ] <- sigma_t_true * decode_r(z[1, ], ls_true)
  for (t in 2:T_full) {
    F[t, ] <- F[t - 1, ] + sigma_t_true * decode_r(z[t, ], ls_true)
  }
  rows <- expand.grid(s_idx = seq_len(L), t_idx = seq_len(T_full))
  rows$s <- dr$grid_coords[rows$s_idx]
  rows$obs_idx <- rows$s_idx
  rows$time_idx <- rows$t_idx
  rows$mu_true <- mapply(function(i, t) F[t, i], rows$s_idx, rows$t_idx)
  rows$y <- rpois(nrow(rows), exp(rows$mu_true))
  list(dr = dr, df = rows, F = F, ls_true = ls_true, sigma_t_true = sigma_t_true)
}

test_that("deepRV_st() captures arguments into a deepRV_st_call", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  spec <- deepRV_st(
    decoder = dr,
    decoder_time = "rw",
    obs_idx = 1:10,
    time_idx = rep(1L, 10),
    ls_prior = prior_uniform(0.05, 0.5),
    sigma_t_prior = prior_exp(1)
  )
  expect_s3_class(spec, "deepRV_st_call")
  expect_identical(spec$decoder_time, "rw")
  expect_output(print(spec), "deepRV_st_call")
})

test_that("deepRV_st() rejects non-rw / Kron decoders", {
  dr_kron <- load_deeprv_kron("unit_square", grid_side = 10,
                               kernel = "matern_1_2")
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  expect_error(
    deepRV_st(decoder = dr_kron, decoder_time = "rw",
              obs_idx = 1:10, time_idx = rep(1L, 10),
              ls_prior = prior_uniform(0.05, 0.5)),
    "Kronecker"
  )
  expect_error(
    deepRV_st(decoder = dr, decoder_time = "ar1",
              obs_idx = 1:10, time_idx = rep(1L, 10),
              ls_prior = prior_uniform(0.05, 0.5)),
    "decoder_time = \"rw\""
  )
})

test_that("deeprv_brm() runs end-to-end on a space-time fixture", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_st_fixture()
  fit <- deeprv_brm(
    y ~ deepRV_st(decoder = fix$dr, obs_idx = obs_idx, time_idx = time_idx,
                  ls_prior = prior_uniform(0.05, 0.5),
                  sigma_t_prior = prior_exp(1)),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L, iter = 400L, warmup = 200L, seed = 1L, cores = 2L
  )
  expect_s3_class(fit, "deeprv_fit")
  expect_identical(fit$by_mode, "st")
  expect_true(all(c("ls", "sigma_t", "F") %in% fit$stanfit@model_pars))
  # F has shape (T, L) per draw, so the extracted array is (S, T, L).
  F_dim <- dim(rstan::extract(fit$stanfit, pars = "F")$F)
  expect_identical(F_dim, c(400L, 5L, as.integer(fix$dr$L)))
})

test_that("posterior_eta_draws() recovers F in the ST model", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_st_fixture()
  fit <- deeprv_brm(
    y ~ deepRV_st(decoder = fix$dr, obs_idx = obs_idx, time_idx = time_idx,
                  ls_prior = prior_uniform(0.05, 0.5),
                  sigma_t_prior = prior_exp(1)),
    data = fix$df, family = poisson(),
    chains = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )
  # R-side forward should match Stan's F draw-by-draw.
  draws <- rstan::extract(fit$stanfit,
                          pars = c("z", "ls", "sigma_t", "b", "F"))
  r_F <- brms.deeprv:::forward_st_batched(fix$dr, draws$z, draws$ls,
                                           draws$sigma_t)
  expect_equal(dim(r_F), dim(draws$F))
  expect_lt(max(abs(r_F - draws$F)), 1e-3)

  # eta = X * b + F[time_idx[n], obs_idx[n]] for each n.
  eta <- posterior_eta_draws(fit)
  b <- as.matrix(draws$b)
  time_idx <- fit$standata$time_idx
  obs_idx <- fit$standata$obs_idx
  expected <- matrix(0, nrow = nrow(draws$F), ncol = length(time_idx))
  for (n in seq_along(time_idx)) {
    expected[, n] <- draws$F[, time_idx[n], obs_idx[n]]
  }
  expected <- b %*% t(fit$X) + expected
  expect_lt(max(abs(eta - expected)), 1e-3)
})
