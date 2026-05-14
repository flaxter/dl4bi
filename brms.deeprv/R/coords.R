# Coordinate-frame helpers. The package never auto-rescales user data;
# users call these explicitly to map their problem into the decoder's
# training domain. See DESIGN.md section 2.3.2.

#' Linearly rescale a 1D coordinate vector to [0, 1]
#'
#' @param s Numeric vector of coordinates.
#' @param from Length-2 numeric c(xmin, xmax) giving the original range.
#' @return Numeric vector with `s` mapped to `[0, 1]` via
#'   `(s - xmin) / (xmax - xmin)`.
#' @export
rescale_to_unit_interval <- function(s, from) {
  if (!is.numeric(s)) stop("`s` must be numeric", call. = FALSE)
  if (!is.numeric(from) || length(from) != 2L || from[1] >= from[2]) {
    stop("`from` must be a length-2 numeric with from[1] < from[2]",
         call. = FALSE)
  }
  (as.numeric(s) - from[1]) / (from[2] - from[1])
}

#' Linearly rescale a 2D coordinate matrix to [0, 1]^2
#'
#' @param s Numeric matrix with two columns (x in col 1, y in col 2).
#' @param from Length-4 numeric c(xmin, xmax, ymin, ymax).
#' @return Matrix the same shape as `s` mapped axis-wise.
#' @export
rescale_to_unit_square <- function(s, from) {
  if (!is.matrix(s) && !is.data.frame(s)) {
    stop("`s` must be a 2-column matrix or data frame", call. = FALSE)
  }
  s <- as.matrix(s)
  if (ncol(s) != 2L) {
    stop("`s` must have exactly two columns (x, y)", call. = FALSE)
  }
  if (!is.numeric(from) || length(from) != 4L ||
      from[1] >= from[2] || from[3] >= from[4]) {
    stop("`from` must be a length-4 numeric ",
         "c(xmin, xmax, ymin, ymax) with xmin < xmax and ymin < ymax",
         call. = FALSE)
  }
  cbind(
    (s[, 1L] - from[1]) / (from[2] - from[1]),
    (s[, 2L] - from[3]) / (from[4] - from[3])
  )
}

#' Find grid-point indices for each observation
#'
#' Returns the 1-based indices in `grid_coords` for each value in `s`,
#' or errors loudly if any observation doesn't sit on the grid within
#' `tol`.
#'
#' @param s Numeric vector of observation coordinates (already in the
#'   decoder's training domain).
#' @param grid_coords The decoder's `grid_coords` vector.
#' @param tol Absolute tolerance for "on the grid" (default 1e-6).
#' @return Integer vector of length `length(s)` with values in
#'   `1..length(grid_coords)`.
#' @export
which_grid_points <- function(s, grid_coords, tol = 1e-6) {
  if (!is.numeric(s) || !is.numeric(grid_coords)) {
    stop("`s` and `grid_coords` must be numeric", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || tol < 0) {
    stop("`tol` must be a non-negative scalar", call. = FALSE)
  }
  s <- as.numeric(s)
  g <- as.numeric(grid_coords)
  # Find nearest grid point for each s by absolute difference.
  out <- integer(length(s))
  diffs <- numeric(length(s))
  for (i in seq_along(s)) {
    d <- abs(g - s[i])
    j <- which.min(d)
    out[i] <- j
    diffs[i] <- d[j]
  }
  bad <- which(diffs > tol)
  if (length(bad) > 0L) {
    stop(sprintf(
      "%d observation(s) don't lie on the decoder grid within tol = %g. ",
      length(bad), tol),
      sprintf("First offender: s[%d] = %g (nearest grid = %g, distance %g). ",
              bad[1L], s[bad[1L]], g[out[bad[1L]]], diffs[bad[1L]]),
      "Use snap_to_grid() to round to the nearest grid point if that's ",
      "what you want.", call. = FALSE)
  }
  out
}

#' Snap observations to nearest grid points (lossy; opt-in)
#'
#' @param s Numeric vector of observation coordinates.
#' @param grid_coords The decoder's `grid_coords` vector.
#' @return A list with `coords` (the snapped coordinates) and `idx`
#'   (the 1-based grid indices).
#' @export
snap_to_grid <- function(s, grid_coords) {
  if (!is.numeric(s) || !is.numeric(grid_coords)) {
    stop("`s` and `grid_coords` must be numeric", call. = FALSE)
  }
  s <- as.numeric(s)
  g <- as.numeric(grid_coords)
  idx <- integer(length(s))
  for (i in seq_along(s)) {
    idx[i] <- which.min(abs(g - s[i]))
  }
  list(coords = g[idx], idx = idx)
}
