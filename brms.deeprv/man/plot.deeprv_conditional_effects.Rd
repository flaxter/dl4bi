\name{plot.deeprv_conditional_effects}
\alias{plot.deeprv_conditional_effects}
\title{Plot a deepRV conditional-effects summary}
\usage{
\method{plot}{deeprv_conditional_effects}(x, ...)
}
\arguments{
\item{x}{A \code{deeprv_conditional_effects} from
\code{\link{conditional_effects}}.}

\item{...}{Forwarded to \code{plot()}.}
}
\value{
The input invisibly so this composes with \code{|>} pipelines.
}
\description{
Uses base graphics so the package doesn't pull \code{ggplot2} as a
hard dependency.
}
