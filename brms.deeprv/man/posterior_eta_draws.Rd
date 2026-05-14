\name{posterior_eta_draws}
\alias{posterior_eta_draws}
\title{Posterior draws of the linear predictor for a deeprv_fit}
\usage{
posterior_eta_draws(fit)
}
\arguments{
\item{fit}{A \code{deeprv_fit} from \code{\link{deeprv_brm}}.}
}
\value{
Matrix \code{(S draws) x (N observations)} on the link scale, equal
to \code{X \%*\% b + mu[obs_idx]}.
}
\description{
Useful as a primitive for custom post-processing; both
\code{\link{posterior_epred.deeprv_fit}} and
\code{\link{posterior_predict.deeprv_fit}} go through this.
}

