# Verifies that the Stan port of MLPDeepRV.decode() matches the JAX outputs
# stored in decoder_artifact.json.
#
# Strategy: rstan::expose_stan_functions compiles the `functions` block of
# mlp_decode.stan into R callables. We call decode() from R with the same
# (z, cond) inputs that the Python export script fed to JAX, and compare
# element-wise.
#
# Run:  Rscript benchmarks/vae/rstan/check_match.R

suppressPackageStartupMessages({
  library(rstan)
  library(jsonlite)
})

here <- function(...) file.path("benchmarks/vae/rstan", ...)
artifact <- fromJSON(here("decoder_artifact.json"), simplifyVector = FALSE)
stopifnot(artifact$arch == "MLPDeepRV")

# Materialize weight matrices from the JSON layer list.
to_matrix <- function(layer) {
  W <- do.call(rbind, lapply(layer$W, unlist))
  b <- unlist(layer$b)
  list(W = W, b = b)
}
L1 <- to_matrix(artifact$layers[[1]])
L2 <- to_matrix(artifact$layers[[2]])

# Expose the Stan `decode` function in this R session.
expose_stan_functions(stanc(file = here("mlp_decode.stan")))

tol <- 1e-5
max_diff <- 0
for (i in seq_along(artifact$test_cases)) {
  case <- artifact$test_cases[[i]]
  z <- unlist(case$z)
  cond <- unlist(case$cond)
  expected <- unlist(case$out)
  got <- decode(z, cond, L1$W, L1$b, L2$W, L2$b)
  d <- max(abs(got - expected))
  max_diff <- max(max_diff, d)
  cat(sprintf("case %d: max|diff| = %.2e\n", i - 1, d))
}
cat(sprintf("overall max|diff| = %.2e\n", max_diff))
if (max_diff >= tol) {
  cat(sprintf("FAIL: exceeds tol %.0e\n", tol))
  quit(status = 1)
}
cat(sprintf("PASS (< %.0e)\n", tol))
