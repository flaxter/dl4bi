test_that("prior_uniform() rejects invalid bounds", {
  expect_error(prior_uniform(0.5, 0.5), "invalid uniform prior")
  expect_error(prior_uniform(0.5, 0.1), "invalid uniform prior")
  expect_error(prior_uniform(-0.1, 0.5), "invalid uniform prior")
  expect_error(prior_uniform("a", "b"), "must be single numeric")
})

test_that("prior_uniform() constructs and prints", {
  p <- prior_uniform(0.05, 0.5)
  expect_s3_class(p, "deepRV_prior")
  expect_identical(p$family, "uniform")
  expect_output(print(p), "uniform")
})

test_that("validate_prior_in_range() catches priors outside trained range", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  expect_error(
    brms.deeprv:::validate_prior_in_range(
      prior_uniform(0.001, 0.5), dr$ls_trained_range, "smoke"),
    "extends outside"
  )
  expect_error(
    brms.deeprv:::validate_prior_in_range(
      prior_uniform(0.05, 1.5), dr$ls_trained_range, "smoke"),
    "extends outside"
  )
  expect_true(brms.deeprv:::validate_prior_in_range(
    prior_uniform(0.05, 0.5), dr$ls_trained_range, "smoke"))
})
