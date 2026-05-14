test_that("load_deeprv() returns a well-formed decoder for a known smoke entry", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  expect_s3_class(dr, "deepRV_decoder")
  expect_identical(dr$domain, "unit_interval")
  expect_identical(as.integer(dr$L), 10L)
  expect_identical(dr$kernel, "matern_1_2")
  expect_length(dr$grid_coords, 10L)
  expect_true(is.matrix(dr$weights$W1))
  expect_equal(dim(dr$weights$W1), c(10L + 1L, ncol(dr$weights$W1)))
  expect_equal(nrow(dr$weights$W2), ncol(dr$weights$W1))
  expect_equal(ncol(dr$weights$W2), 10L)
})

test_that("load_deeprv() errors with a useful listing on cache miss", {
  err <- expect_error(
    load_deeprv("unit_interval", grid_size = 99, kernel = "rbf"),
    "no decoder for"
  )
  # Should at least name one of the shipped fixtures in the available list.
  expect_match(conditionMessage(err), "unit_interval_10_matern_1_2", fixed = TRUE)
})

test_that("load_deeprv() rejects unit_square and points at the kron helper", {
  expect_error(
    load_deeprv("unit_square", grid_size = 10, kernel = "matern_1_2"),
    "load_deeprv_kron"
  )
})

test_that("load_deeprv() rejects unsupported kernels and domains", {
  expect_error(
    load_deeprv("unit_interval", grid_size = 10, kernel = "periodic"),
    "kernel=periodic not supported"
  )
  expect_error(
    load_deeprv("hyperbolic", grid_size = 10, kernel = "rbf"),
    "domain=hyperbolic not supported"
  )
})

test_that("validate_decoder() catches a tampered fingerprint", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  dr$fingerprint <- paste0(substr(dr$fingerprint, 1, 60), "deadbeef")
  expect_error(
    brms.deeprv:::validate_decoder(dr, source = "tampered"),
    "fingerprint mismatch"
  )
})

test_that("validate_decoder() catches a tampered ls_trained_range", {
  # If the .rds was edited to claim a wider ls range than it was trained
  # for, the fingerprint recompute should diverge.
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  dr$ls_trained_range <- c(0.01, 2.0)
  expect_error(brms.deeprv:::validate_decoder(dr), "fingerprint mismatch")
})

test_that("load_deeprv_kron() composes a 2D decoder from two 1D loads", {
  kron <- load_deeprv_kron("unit_square", grid_side = 10, kernel = "matern_3_2")
  expect_s3_class(kron, "deepRV_decoder_kron")
  expect_s3_class(kron, "deepRV_decoder")
  expect_identical(kron$domain, "unit_square")
  expect_identical(as.integer(kron$grid_side), 10L)
  expect_identical(as.integer(kron$L), 100L)
  expect_equal(dim(kron$grid_coords), c(100L, 2L))
  expect_s3_class(kron$x_decoder, "deepRV_decoder")
  expect_s3_class(kron$y_decoder, "deepRV_decoder")
})

test_that("print methods don't error", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  kron <- load_deeprv_kron("unit_square", grid_side = 10, kernel = "matern_3_2")
  expect_output(print(dr), "deepRV_decoder")
  expect_output(print(kron), "deepRV_decoder_kron")
})

test_that("fingerprint formula agrees with the Python convention (DESIGN.md §4.8)", {
  # Hard-coded fingerprint for the smoke matern_1_2 decoder. If Python's
  # train_decoders.py ever drifts from R's %.8f formatting, this will
  # fail before any decoder loads.
  fp <- brms.deeprv:::deeprv_fingerprint(
    "MLPDeepRV", "smoke.0", "unit_interval", 10L, "matern_1_2",
    0.01, 1.0
  )
  expect_identical(
    fp,
    "f3e712b95cfc86afddc8bd4dead7ec9e35004d836fc7dd9c186aa8ffc5bd5f19"
  )
})
