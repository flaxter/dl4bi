\name{deeprv_brm}
\alias{deeprv_brm}
\title{Fit a deepRV-augmented brms-style model}
\usage{
deeprv_brm(formula, data, family = stats::poisson(),
           chains = 2L, iter = 1000L, warmup = floor(iter/2),
           seed = NULL, cores = getOption("mc.cores", 1L),
           control = list(adapt_delta = 0.95), ...)
}
\arguments{
\item{formula}{A formula of the form
\code{y ~ deepRV(s, decoder = ..., obs_idx = ..., ls_prior = ...)}.}

\item{data}{A data frame containing the response and any columns
referenced inside \code{deepRV(...)}.}

\item{family}{A family object. Only \code{poisson()} is supported in MVP.}

\item{chains, iter, warmup, seed, cores, control}{Forwarded to
\code{rstan::stan()}.}

\item{...}{Caught and rejected for now (no \code{brms} passthrough yet).}
}
\value{
A \code{deeprv_fit} object containing the \code{stanfit}, the generated
Stan code, the standata, and the parsed \code{deepRV_call}.
}
\description{
MVP scope:
\itemize{
  \item Exactly one \code{deepRV()} term in the formula.
  \item Intercept-only fixed effects (no covariates other than the
    decoder term).
  \item Poisson family.
  \item \code{\link{prior_uniform}} on the length-scale.
}
Each of these restrictions is checked at the top of \code{deeprv_brm()}
and lifts in a follow-on commit.
}
