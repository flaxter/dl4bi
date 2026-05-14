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
\code{y ~ <covariates> + deepRV(s, decoder = ..., obs_idx = ...,
ls_prior = ...)}. RHS covariates are passed through
\code{model.matrix()} with the usual rules (intercept on by default).}

\item{data}{A data frame containing the response, any RHS columns,
and any columns referenced inside \code{deepRV(...)}.}

\item{family}{A family object. \code{poisson()} and \code{gaussian()}
are supported.}

\item{chains, iter, warmup, seed, cores, control}{Forwarded to
\code{rstan::stan()}.}

\item{...}{Caught and rejected for now (no \code{brms} passthrough yet).}
}
\value{
A \code{deeprv_fit} object containing the \code{stanfit}, the
generated Stan code, the standata, the parsed \code{deepRV_call}, and
the design matrix \code{X} (for downstream prediction).
}
\description{
Supported v0.1 surface:
\itemize{
  \item Exactly one \code{deepRV()} term in the formula.
  \item Any RHS covariates handled via \code{model.matrix(~ rhs, data)}.
    Coefficient vector \code{b} gets a default N(0, 1) prior.
  \item \code{family} in \code{poisson()} or \code{gaussian()}.
  \item \code{\link{prior_uniform}} on the length-scale.
}
The \code{by} / \code{gr} / \code{ls_pooling} knobs are reserved for
v0.2 and error if non-default.
}

