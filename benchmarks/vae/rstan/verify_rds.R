# Forward-match check on a packed .rds decoder.
#
# Loads <decoder>.rds, exposes mlp_decode.stan's `decode` function via
# rstan::expose_stan_functions, and compares Stan's forward output
# against the JAX-generated test cases stored inside the .rds. This is
# the test described in DESIGN.md §4.8 ("forward-pass match is the most
# important test").
#
# Usage:
#   Rscript benchmarks/vae/rstan/verify_rds.R <path/to/decoder.rds>

suppressPackageStartupMessages({
  library(rstan)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript verify_rds.R <decoder.rds>")
rds_path <- args[1]
stopifnot(file.exists(rds_path))

dr <- readRDS(rds_path)
stopifnot(dr$arch == "MLPDeepRV")
cat(sprintf("Loaded %s: domain=%s L=%d kernel=%s arch_version=%s\n",
            basename(rds_path), dr$domain, dr$L, dr$kernel, dr$arch_version))

stan_file <- file.path("benchmarks/vae/rstan", "mlp_decode.stan")
expose_stan_functions(stanc(file = stan_file))

W1 <- dr$weights$W1; b1 <- dr$weights$b1
W2 <- dr$weights$W2; b2 <- dr$weights$b2

tol <- 1e-5
max_diff <- 0
for (i in seq_along(dr$test_cases)) {
  case <- dr$test_cases[[i]]
  got  <- decode(case$z, case$cond, W1, b1, W2, b2)
  d    <- max(abs(got - case$out))
  max_diff <- max(max_diff, d)
  cat(sprintf("  case %d: max|diff| = %.2e\n", i - 1, d))
}
cat(sprintf("overall max|diff| = %.2e\n", max_diff))
if (max_diff >= tol) {
  cat(sprintf("FAIL: exceeds tol %.0e\n", tol))
  quit(status = 1)
}
cat(sprintf("PASS (< %.0e)\n", tol))
