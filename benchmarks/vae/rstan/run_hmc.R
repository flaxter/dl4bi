# Sample from the full RStan model with the (untrained) MLPDeepRV decoder
# as the spatial prior, on synthetic Poisson data generated from that same
# decoder at a known truth (see export_mlp_decoder.py).
#
# This is a wiring test — not a calibration claim — since the decoder weights
# are random (no training). Posterior recovery of (beta, ls) will be loose:
# the goal is "Stan accepts the model, gradients evaluate, HMC samples without
# error, and the divergences/rhat behave."
#
# Run:  Rscript benchmarks/vae/rstan/run_hmc.R

suppressPackageStartupMessages({
  library(rstan)
  library(jsonlite)
})

options(mc.cores = 2)
rstan_options(auto_write = TRUE)

here <- function(...) file.path("benchmarks/vae/rstan", ...)
artifact <- fromJSON(here("decoder_artifact.json"), simplifyVector = FALSE)
stopifnot(artifact$arch == "MLPDeepRV")

to_matrix <- function(layer) {
  W <- do.call(rbind, lapply(layer$W, unlist))
  list(W = W, b = unlist(layer$b))
}
L1 <- to_matrix(artifact$layers[[1]])
L2 <- to_matrix(artifact$layers[[2]])

stan_data <- list(
  L = artifact$L,
  cond_dim = artifact$cond_dim,
  hidden = artifact$dims[[1]],
  W1 = L1$W, b1 = L1$b,
  W2 = L2$W, b2 = L2$b,
  y = as.integer(unlist(artifact$inference$y)),
  obs_mask = as.integer(unlist(artifact$inference$obs_mask))
)
truth <- artifact$inference$truth

cat(sprintf("L=%d, n_obs=%d, truth: ls*=%.3f beta*=%+.3f\n",
            stan_data$L, sum(stan_data$obs_mask),
            truth$ls, truth$beta))

t0 <- Sys.time()
fit <- stan(
  file = here("mlp_decode.stan"),
  data = stan_data,
  chains = 2,
  iter = 1000,
  warmup = 500,
  seed = 1,
  refresh = 0,
  control = list(adapt_delta = 0.95)
)
wall <- as.numeric(Sys.time() - t0, units = "secs")

print(fit, pars = c("beta", "ls"), probs = c(0.05, 0.5, 0.95))

post <- rstan::extract(fit, pars = c("beta", "ls"))
ci <- function(x) sprintf("[%.3f, %.3f]", quantile(x, 0.05), quantile(x, 0.95))
cat(sprintf("\nbeta:  truth=%+.3f   posterior mean=%+.3f   90%% CI %s\n",
            truth$beta, mean(post$beta), ci(post$beta)))
cat(sprintf("ls:    truth=%+.3f   posterior mean=%+.3f   90%% CI %s\n",
            truth$ls, mean(post$ls), ci(post$ls)))

sp <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
ndiv <- sum(sapply(sp, function(s) sum(s[, "divergent__"])))
cat(sprintf("\ndivergences (post-warmup): %d\n", ndiv))
cat(sprintf("wall time (sampling + compile): %.1fs\n", wall))
