\name{posterior_predict.deeprv_fit}
\alias{posterior_predict.deeprv_fit}
\title{Posterior predictive draws for a deeprv_fit}
\usage{
\method{posterior_predict}{deeprv_fit}(object, newdata = NULL, ...)
}
\arguments{
\item{object}{A \code{deeprv_fit} from \code{\link{deeprv_brm}}.}

\item{newdata}{Must be \code{NULL} in v0.1.}

\item{...}{Unused.}
}
\value{
Matrix \code{(S draws) x (N observations)} of posterior predictive
draws on the response scale, sampled from the family.
}
\description{
S3 method for \code{rstantools::posterior_predict()} dispatching on
\code{deeprv_fit}.
}
