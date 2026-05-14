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
    set.seed(101L)
    L <- as.integer(dr$L)
    y <- rpois(L, 1)
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

test_that("conditional_effects() returns a grid-indexed summary", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  ce <- conditional_effects(fit)
  expect_s3_class(ce, "deeprv_conditional_effects")
  expect_length(ce, 1L)
  df <- ce$deepRV
  expect_equal(nrow(df), as.integer(fit$decoder$L))
  expect_setequal(names(df),
                  c("grid_index", "s", "estimate", "lower", "upper"))
  # CI bounds bracket the mean (or coincide, in degenerate cases).
  expect_true(all(df$lower <= df$estimate))
  expect_true(all(df$estimate <= df$upper))
})

test_that("conditional_effects() honors a custom probs argument", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  ce_50 <- conditional_effects(fit, probs = c(0.25, 0.75))
  ce_90 <- conditional_effects(fit, probs = c(0.05, 0.95))
  # The 90% intervals must be at least as wide as the 50% intervals.
  expect_true(all(ce_90$deepRV$upper >= ce_50$deepRV$upper))
  expect_true(all(ce_90$deepRV$lower <= ce_50$deepRV$lower))
})

test_that("conditional_effects() rejects malformed probs", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  expect_error(conditional_effects(fit, probs = 0.5), "length 2")
  expect_error(conditional_effects(fit, probs = c(0.9, 0.1)),
               "probs\\[1\\] < probs\\[2\\]")
  expect_error(conditional_effects(fit, probs = c(-0.1, 0.5)),
               "0 <= probs")
})

test_that("plot.deeprv_conditional_effects() runs without error", {
  skip_if_no_rstan()
  skip_on_cran()

  fit <- make_tiny_fit()
  ce <- conditional_effects(fit)
  pdf(file = NULL)
  on.exit(dev.off(), add = TRUE)
  expect_invisible(plot(ce))
})
