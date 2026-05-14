\name{deepRV_st}
\alias{deepRV_st}
\title{Mark a deepRV space-time term in a deeprv_brm() formula}
\usage{
deepRV_st(decoder, decoder_time = "rw", obs_idx, time_idx, ls_prior,
          sigma_t_prior = prior_exp(1), ls_t_prior = NULL)
}
\arguments{
\item{decoder}{A 1D \code{deepRV_decoder} for the spatial axis (from
\code{\link{load_deeprv}}). Kronecker spatial decoders are reserved for
a follow-up.}

\item{decoder_time}{One of: \code{"rw"} (random walk; default),
\code{"ar1"} (lag-1 autoregressive with stationary marginal,
\code{phi ~ uniform(-1, 1)}), or a 1D \code{deepRV_decoder} for time
(applies the decoder column-wise along the time axis).}

\item{obs_idx}{Integer vector of length \code{nrow(data)} mapping each
observation to its 1-based spatial grid index.}

\item{time_idx}{Integer vector of length \code{nrow(data)} mapping each
observation to its 1-based time index.}

\item{ls_prior}{A \code{deepRV_prior} for the spatial length-scale.}

\item{sigma_t_prior}{A \code{deepRV_prior} for the random-walk or
AR1 innovation scale (default \code{prior_exp(1)}). Ignored when
\code{decoder_time} is a \code{deepRV_decoder}, since the time
decoder bakes the amplitude into its weights during training.}

\item{ls_t_prior}{A \code{deepRV_prior} for the time-decoder's
length-scale. Required when \code{decoder_time} is a
\code{deepRV_decoder}; ignored otherwise.}
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

