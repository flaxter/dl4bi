\name{deepRV}
\alias{deepRV}
\title{Mark a deepRV decoder term in a deeprv_brm() formula}
\usage{
deepRV(s, decoder, obs_idx, ls_prior, by = NULL, gr = FALSE,
       ls_pooling = "complete")
}
\arguments{
\item{s}{Coordinate column from \code{data}. Used for diagnostics only
in v0.1; \code{obs_idx} is what actually drives indexing into the
decoder's grid.}

\item{decoder}{A \code{deepRV_decoder} from \code{\link{load_deeprv}}.
Required.}

\item{obs_idx}{Integer vector of length \code{nrow(data)} mapping each
observation to its 1-based grid index in \code{decoder$grid_coords}.
Required.}

\item{ls_prior}{A \code{deepRV_prior} from \code{\link{prior_uniform}}.
Required. Must lie inside \code{decoder$ls_trained_range}.}

\item{by, gr, ls_pooling}{Reserved for v0.1 follow-on features. Must be at
their defaults in the MVP; non-default values error.}
}
\value{
A \code{deepRV_call} object capturing its arguments. Consumed by
\code{\link{deeprv_brm}}.
}
\description{
\code{deepRV()} doesn't compute anything on its own --- it captures
its arguments so \code{\link{deeprv_brm}} can recognise the term
during formula parsing and translate it into the appropriate Stan
additions.
}
