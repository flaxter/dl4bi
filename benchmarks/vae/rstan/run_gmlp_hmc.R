# HMC with trained gMLPDeepRV(num_blks=2) as the spatial prior. Mirrors
# run_hmc.R but builds standata from gmlp_decoder_artifact.json. Reports
# (beta, ls) recovery, divergences, Rhat, n_eff, and wall time so we can
# compare against the MLP run.

suppressPackageStartupMessages({
  library(rstan)
  library(jsonlite)
})

options(mc.cores = 2)
rstan_options(auto_write = TRUE)

here <- function(...) file.path("benchmarks/vae/rstan", ...)
art <- fromJSON(here("gmlp_decoder_artifact.json"), simplifyVector = FALSE)
stopifnot(art$arch == "gMLPDeepRV")

as_mat <- function(x) do.call(rbind, lapply(x, unlist))
W <- list()
for (nm in names(art$weights)) {
  v <- art$weights[[nm]]
  # as.array() preserves the 1D shape so Stan reads length-1 vectors (e.g.
  # head_b1) as vector[1] rather than coercing to a bare scalar.
  W[[nm]] <- if (is.list(v[[1]])) as_mat(v) else as.array(unlist(v))
}
s_mat <- as_mat(art$s_mat)

stan_data <- c(list(
  L = art$L, cond_dim = art$cond_dim, s_mat = s_mat,
  y = as.integer(unlist(art$inference$y)),
  obs_mask = as.integer(unlist(art$inference$obs_mask))
), W)
truth <- art$inference$truth

cat(sprintf("L=%d, n_obs=%d, truth: ls*=%.3f beta*=%+.3f\n",
            stan_data$L, sum(stan_data$obs_mask), truth$ls, truth$beta))

t0 <- Sys.time()
fit <- stan(
  file = here("gmlp_decode.stan"),
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
