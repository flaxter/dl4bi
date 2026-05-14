\name{prior_exp}
\alias{prior_exp}
\title{Exponential prior for non-negative scalar parameters}
\usage{
prior_exp(rate)
}
\arguments{
\item{rate}{Positive scalar; the exponential rate parameter.}
}
\value{
A \code{deepRV_prior} object.
}
\description{
Used as the default prior for the random-walk innovation scale
\code{sigma_t} in \code{\link{deepRV_st}}.
}
