\name{conditional_effects}
\alias{conditional_effects}
\alias{conditional_effects.deeprv_fit}
\title{Posterior summaries of a deepRV spatial field}
\usage{
conditional_effects(x, probs = c(0.05, 0.95), ...)

\method{conditional_effects}{deeprv_fit}(x, probs = c(0.05, 0.95), ...)
}
\arguments{
\item{x}{A \code{deeprv_fit} from \code{\link{deeprv_brm}}.}

\item{probs}{Numeric pair giving the credible interval (default
90 percent).}

\item{...}{Reserved for future arguments.}
}
\value{
A \code{deeprv_conditional_effects} object: a list of data frames,
one per deepRV term (currently always one). Each data frame has
columns \code{grid_index}, \code{s} (the coordinate from
\code{decoder$grid_coords}), \code{estimate} (posterior mean),
\code{lower}, and \code{upper}.
}
\description{
Returns the posterior mean and credible-interval bounds of the
decoder output \code{mu} evaluated at every training grid point.
Mirrors \code{brms::conditional_effects()} for \code{brmsfit}
objects.
}
\seealso{
\code{\link{plot.deeprv_conditional_effects}} for the base-graphics
plot method.
}
