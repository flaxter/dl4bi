\name{deepRV_st}
\alias{deepRV_st}
\title{Mark a deepRV space-time term in a deeprv_brm() formula}
\usage{
deepRV_st(decoder, decoder_time = "rw", obs_idx, time_idx, ls_prior,
          sigma_t_prior = prior_exp(1))
}
\arguments{
\item{decoder}{A 1D \code{deepRV_decoder} for the spatial axis (from
\code{\link{load_deeprv}}). Kronecker spatial decoders are reserved for
a follow-up.}

\item{decoder_time}{One of: \code{"rw"} (random walk in time; default).
\code{"ar1"} and decoder-based time priors are reserved for a follow-up.}

\item{obs_idx}{Integer vector of length \code{nrow(data)} mapping each
observation to its 1-based spatial grid index.}

\item{time_idx}{Integer vector of length \code{nrow(data)} mapping each
observation to its 1-based time index.}

\item{ls_prior}{A \code{deepRV_prior} for the spatial length-scale.}

\item{sigma_t_prior}{A \code{deepRV_prior} for the random-walk
innovation scale (default \code{prior_exp(1)}).}
}
\value{
A \code{deepRV_st_call} object capturing its arguments. Consumed by
\code{\link{deeprv_brm}}.
}
\description{
Like \code{\link{deepRV}} but with an additional time axis. The Stan
model has a \code{matrix[T, L] z} latent; the field
\code{F[t, :] = F[t-1, :] + sigma_t * decode(z[t, :], ls)} is a random
walk in time over the spatial smooth.
}

