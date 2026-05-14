# Same HMC as run_gmlp_hmc.R, but the `decode` function is implemented in
# templated C++ (gmlp_decode_impl.hpp) and linked via `--allow-undefined`.
# Compares per-iter cost against the all-Stan version.

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

impl_hpp <- normalizePath(here("gmlp_decode_impl.hpp"))
cat(sprintf("Linking external C++: %s\n", impl_hpp))

t_compile_start <- Sys.time()
mod <- stan_model(
  file = here("gmlp_decode_extc.stan"),
  allow_undefined = TRUE,
  includes = paste0('\n#include "', impl_hpp, '"\n')
)
t_compile <- as.numeric(Sys.time() - t_compile_start, units = "secs")
cat(sprintf("Compile time: %.1fs\n", t_compile))

t_sample_start <- Sys.time()
fit <- sampling(
  mod,
  data = stan_data,
  chains = 2,
  iter = 1000,
  warmup = 500,
  seed = 1,
  refresh = 0,
  control = list(adapt_delta = 0.95)
)
t_sample <- as.numeric(Sys.time() - t_sample_start, units = "secs")
wall <- t_compile + t_sample

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
cat(sprintf("compile: %.1fs   sampling: %.1fs   total: %.1fs\n",
            t_compile, t_sample, wall))
