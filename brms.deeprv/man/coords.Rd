\name{coords}
\alias{rescale_to_unit_interval}
\alias{rescale_to_unit_square}
\alias{which_grid_points}
\alias{snap_to_grid}
\title{Coordinate-frame helpers for deepRV decoders}
\usage{
rescale_to_unit_interval(s, from)

rescale_to_unit_square(s, from)

which_grid_points(s, grid_coords, tol = 1e-6)

snap_to_grid(s, grid_coords)
}
\arguments{
\item{s}{Numeric vector (1D) or two-column matrix (2D) of observation
coordinates.}

\item{from}{For \code{rescale_to_unit_interval()}, a length-2 numeric
\code{c(xmin, xmax)}. For \code{rescale_to_unit_square()}, a length-4
\code{c(xmin, xmax, ymin, ymax)}.}

\item{grid_coords}{The decoder's \code{grid_coords} vector.}

\item{tol}{Absolute tolerance for "on the grid"
(default \code{1e-6}).}
}
\value{
\describe{
  \item{\code{rescale_to_unit_interval}}{Numeric vector mapped to
    \code{[0, 1]} via \code{(s - xmin) / (xmax - xmin)}.}
  \item{\code{rescale_to_unit_square}}{Numeric matrix the same shape
    as \code{s} mapped axis-wise to \code{[0, 1]^2}.}
  \item{\code{which_grid_points}}{Integer vector of 1-based indices
    in \code{grid_coords}. Errors if any observation is off-grid by
    more than \code{tol}.}
  \item{\code{snap_to_grid}}{List with \code{coords} (the snapped
    coordinates) and \code{idx} (the 1-based grid indices).}
}
}
\description{
The package never auto-rescales user data; users call these
explicitly to map their problem into the decoder's training domain.
See \code{DESIGN.md} section 2.3.2.
}

