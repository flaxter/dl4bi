make_fixture <- function() {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  df <- data.frame(s = dr$grid_coords, y = rep(1L, dr$L))
  df$obs_idx <- seq_len(nrow(df))
  list(dr = dr, df = df)
}

test_that("parse_deeprv_formula() pulls a clean spec from y ~ deepRV(...)", {
  fix <- make_fixture()
  parsed <- brms.deeprv:::parse_deeprv_formula(
    y ~ deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
               ls_prior = prior_uniform(0.05, 0.5)),
    data = fix$df
  )
  expect_length(parsed$deepRV, 1L)
  expect_s3_class(parsed$deepRV[[1L]], "deepRV_call")
  expect_null(parsed$rhs)
  expect_identical(parsed$lhs, as.name("y"))
})

test_that("parse_deeprv_formula() retains other RHS terms as residual", {
  fix <- make_fixture()
  fix$df$x1 <- rnorm(nrow(fix$df))
  fix$df$x2 <- rnorm(nrow(fix$df))
  parsed <- brms.deeprv:::parse_deeprv_formula(
    y ~ x1 + x2 + deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
                          ls_prior = prior_uniform(0.05, 0.5)),
    data = fix$df
  )
  expect_length(parsed$deepRV, 1L)
  expect_false(is.null(parsed$rhs))
  expect_match(deparse(parsed$rhs), "x1.*x2", all = FALSE)
})

test_that("parse_deeprv_formula() errors on zero or multiple deepRV terms", {
  fix <- make_fixture()
  expect_error(
    brms.deeprv:::parse_deeprv_formula(y ~ 1, data = fix$df),
    "no deepRV"
  )
  expect_error(
    brms.deeprv:::parse_deeprv_formula(
      y ~ deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
                 ls_prior = prior_uniform(0.05, 0.5)) +
          deepRV(s, decoder = fix$dr, obs_idx = obs_idx,
                 ls_prior = prior_uniform(0.05, 0.5)),
      data = fix$df
    ),
    "multiple deepRV"
  )
})

test_that("deepRV() captures args into a deepRV_call object", {
  fix <- make_fixture()
  spec <- deepRV(s, decoder = fix$dr, obs_idx = fix$df$obs_idx,
                 ls_prior = prior_uniform(0.05, 0.5))
  expect_s3_class(spec, "deepRV_call")
  expect_identical(spec$ls_pooling, "complete")
  expect_false(spec$gr)
  expect_null(spec$by)
  expect_output(print(spec), "deepRV_call")
})
