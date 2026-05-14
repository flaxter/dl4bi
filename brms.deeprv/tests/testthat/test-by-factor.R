skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

# Synthetic 2-group Poisson fixture: each group gets its own z but the
# shared decoder and ls. By construction the per-group spatial fields
# are different.
make_two_group_fixture <- function(seed = 31L) {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  set.seed(seed)
  L <- as.integer(dr$L)
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  ls_true <- 0.2
  decode_r <- function(z) {
    h <- pmax(crossprod(W1, c(z, ls_true)) + b1, 0)
    as.numeric(crossprod(W2, h) + b2)
  }
  mu_a <- decode_r(rnorm(L))
  mu_b <- decode_r(rnorm(L))
  # Two groups, L observations each, with group-specific spatial field.
  rows <- expand.grid(s_idx = seq_len(L), group = c("a", "b"),
                       KEEP.OUT.ATTRS = FALSE,
                       stringsAsFactors = FALSE)
  rows$group <- factor(rows$group, levels = c("a", "b"))
  rows$s <- dr$grid_coords[rows$s_idx]
  rows$obs_idx <- rows$s_idx
  rows$mu_true <- ifelse(rows$group == "a", mu_a[rows$s_idx],
                                            mu_b[rows$s_idx])
  rows$y <- rpois(nrow(rows), exp(rows$mu_true))
  list(dr = dr, df = rows, mu_a = mu_a, mu_b = mu_b)
}

test_that("classify_by() routes factor to factor mode and char to factor mode", {
  expect_identical(brms.deeprv:::classify_by(NULL)$mode, "none")
  expect_identical(brms.deeprv:::classify_by(c(0.1, 0.2))$mode, "svc")
  f <- factor(c("a", "b", "a"))
  info_f <- brms.deeprv:::classify_by(f)
  expect_identical(info_f$mode, "factor")
  expect_identical(info_f$values, c(1L, 2L, 1L))
  expect_identical(info_f$levels, c("a", "b"))

  info_chr <- brms.deeprv:::classify_by(c("a", "b", "a"))
  expect_identical(info_chr$mode, "factor")
  expect_identical(info_chr$values, c(1L, 2L, 1L))
})

test_that("deeprv_brm() with by = <factor> runs end-to-end", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data    = fix$df,
    family  = poisson(),
    chains  = 2L,
    iter    = 400L,
    warmup  = 200L,
    seed    = 1L,
    cores   = 2L
  )
  expect_identical(fit$by_mode, "factor")
  expect_identical(fit$n_groups, 2L)
  expect_identical(fit$by_levels, c("a", "b"))
  expect_identical(fit$standata$G, 2L)
  expect_length(fit$standata$group_idx, nrow(fix$df))

  # Stan now exposes mu as a (G, L) matrix.
  mu_dim <- dim(rstan::extract(fit$stanfit, pars = "mu")$mu)
  expect_identical(mu_dim, c(400L, 2L, as.integer(fix$dr$L)))

  # No divergences (loose ceiling for smoke decoder).
  sp <- rstan::get_sampler_params(fit$stanfit, inc_warmup = FALSE)
  ndiv <- sum(sapply(sp, function(s) sum(s[, "divergent__"])))
  expect_lt(ndiv, 20L)
})

test_that("posterior_eta_draws() gathers per-group mu via group_idx", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data = fix$df, family = poisson(),
    chains = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )
  eta <- posterior_eta_draws(fit)
  draws <- rstan::extract(fit$stanfit, pars = c("z", "ls", "b", "mu"))
  b <- as.matrix(draws$b)

  group_idx <- fit$standata$group_idx
  obs_idx <- fit$standata$obs_idx
  N <- length(group_idx)
  # Recompute eta from Stan's mu directly for cross-check.
  expected <- matrix(0, nrow = nrow(draws$mu), ncol = N)
  for (n in seq_len(N)) {
    expected[, n] <- draws$mu[, group_idx[n], obs_idx[n]]
  }
  expected <- b %*% t(fit$X) + expected
  expect_lt(max(abs(eta - expected)), 1e-3)
})

test_that("conditional_effects() returns per-group rows", {
  skip_if_no_rstan()
  skip_on_cran()

  fix <- make_two_group_fixture()
  fit <- deeprv_brm(
    y ~ deepRV(s, by = group, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data = fix$df, family = poisson(),
    chains = 1L, iter = 100L, warmup = 50L, seed = 1L, cores = 1L
  )
  ce <- conditional_effects(fit)
  df <- ce$deepRV
  expect_true("group" %in% names(df))
  expect_equal(nrow(df), 2L * as.integer(fix$dr$L))
  expect_setequal(unique(df$group), c("a", "b"))
})
