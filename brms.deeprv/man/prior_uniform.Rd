\name{prior_uniform}
\alias{prior_uniform}
\title{Uniform prior over the decoder's length-scale parameter}
\usage{
prior_uniform(lower, upper)
}
\arguments{
\item{lower, upper}{Bounds. \code{lower < upper}, both finite, both > 0.}
}
\value{
A \code{deepRV_prior} object consumed by \code{\link{deeprv_brm}}.
}
\description{
Returned object is rendered into the generated Stan program as a
parameter bound on \code{ls}. Must lie inside the decoder's
\code{ls_trained_range}; \code{\link{deeprv_brm}} errors otherwise
(see \code{DESIGN.md} section 2.3.3).
}
