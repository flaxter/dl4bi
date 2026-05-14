\name{load_deeprv_kron}
\alias{load_deeprv_kron}
\title{Load a Kronecker-composed 2D decoder for unit_square}
\usage{
load_deeprv_kron(domain, grid_side, kernel)
}
\arguments{
\item{domain}{Must be \code{"unit_square"}.}

\item{grid_side}{Number of points per axis; the full grid has
\code{grid_side^2} cells.}

\item{kernel}{One of \code{"matern_1_2"}, \code{"matern_3_2"},
\code{"matern_5_2"}, or \code{"rbf"}; applied identically along
both axes.}
}
\value{
A list with class \code{c("deepRV_decoder_kron", "deepRV_decoder")}
following the schema in \code{DESIGN.md} section 2.5.
\code{$x_decoder} and \code{$y_decoder} hold the 1D decoders;
\code{$grid_coords} is the full 2D grid as a \code{grid_side^2 x 2}
matrix.
}
\description{
Pairs two 1D \code{unit_interval} decoders (one along x, one along y).
Sampling reduces to \code{F = L_x \%*\% Z \%*\% t(L_y)} at fit time.
Only axis-wise separable kernels are valid in v0.1.
}

