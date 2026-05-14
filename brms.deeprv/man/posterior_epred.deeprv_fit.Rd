\name{posterior_epred.deeprv_fit}
\alias{posterior_epred.deeprv_fit}
\title{Posterior expected response for a deeprv_fit}
\usage{
\method{posterior_epred}{deeprv_fit}(object, newdata = NULL, ...)
}
\arguments{
\item{object}{A \code{deeprv_fit} from \code{\link{deeprv_brm}}.}

\item{newdata}{Must be \code{NULL} in v0.1 - new-location prediction
is not supported with \code{MLPDeepRV} (see \code{DESIGN.md} section
2.7.1).}

\item{...}{Unused.}
}
\value{
Matrix \code{(S draws) x (N observations)} of posterior draws of the
expected response on the response scale (e.g. \code{exp(eta)} for
Poisson, \code{eta} for Gaussian).
}
\description{
S3 method for \code{rstantools::posterior_epred()} dispatching on
\code{deeprv_fit}.
}
