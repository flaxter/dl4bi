test_that("rescale_to_unit_interval() maps endpoints correctly", {
  expect_equal(rescale_to_unit_interval(c(0, 5, 10), from = c(0, 10)),
               c(0, 0.5, 1))
  expect_equal(rescale_to_unit_interval(c(-3, 2, 7), from = c(-3, 7)),
               c(0, 0.5, 1))
})

test_that("rescale_to_unit_interval() rejects malformed from", {
  expect_error(rescale_to_unit_interval(0.5, from = 0), "length-2 numeric")
  expect_error(rescale_to_unit_interval(0.5, from = c(1, 0)), "from\\[1\\] < from\\[2\\]")
  expect_error(rescale_to_unit_interval("abc", from = c(0, 1)), "must be numeric")
})

test_that("rescale_to_unit_square() maps both axes independently", {
  pts <- matrix(c(0, 5, 10,
                  -1, 0, 1),
                ncol = 2L)
  out <- rescale_to_unit_square(pts, from = c(0, 10, -1, 1))
  expect_equal(out[, 1], c(0, 0.5, 1))
  expect_equal(out[, 2], c(0, 0.5, 1))
})

test_that("rescale_to_unit_square() rejects wrong shape", {
  expect_error(rescale_to_unit_square(c(0.1, 0.2), from = c(0, 1, 0, 1)),
               "2-column matrix")
  expect_error(rescale_to_unit_square(matrix(0, 3, 3), from = c(0, 1, 0, 1)),
               "exactly two columns")
  expect_error(rescale_to_unit_square(matrix(0, 3, 2), from = c(0, 1, 1, 0)),
               "ymin < ymax")
})

test_that("which_grid_points() finds exact matches", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  # All grid points map back to themselves.
  idx <- which_grid_points(dr$grid_coords, dr$grid_coords)
  expect_identical(idx, seq_along(dr$grid_coords))
})

test_that("which_grid_points() tolerates small numerical drift", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  drifted <- dr$grid_coords + 1e-10
  idx <- which_grid_points(drifted, dr$grid_coords, tol = 1e-6)
  expect_identical(idx, seq_along(dr$grid_coords))
})

test_that("which_grid_points() errors clearly off-grid", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  off <- c(dr$grid_coords[1L], 0.123456789, dr$grid_coords[3L])
  expect_error(which_grid_points(off, dr$grid_coords, tol = 1e-6),
               "don't lie on the decoder grid")
  expect_error(which_grid_points(off, dr$grid_coords, tol = 1e-6),
               "snap_to_grid")
})

test_that("snap_to_grid() returns nearest grid point and its index", {
  dr <- load_deeprv("unit_interval", grid_size = 10, kernel = "matern_1_2")
  out <- snap_to_grid(c(0.0, 0.123, 0.999), dr$grid_coords)
  expect_length(out$coords, 3L)
  expect_length(out$idx, 3L)
  expect_true(all(out$idx >= 1L & out$idx <= dr$L))
  expect_equal(out$coords, dr$grid_coords[out$idx])
})
