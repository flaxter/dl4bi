# Verifies the Stan port of gMLPDeepRV.decode() matches JAX outputs
# in gmlp_decoder_artifact.json.

suppressPackageStartupMessages({
  library(rstan)
  library(jsonlite)
})

here <- function(...) file.path("benchmarks/vae/rstan", ...)
art <- fromJSON(here("gmlp_decoder_artifact.json"), simplifyVector = FALSE)
stopifnot(art$arch == "gMLPDeepRV")

as_mat <- function(x) do.call(rbind, lapply(x, unlist))
as_vec <- function(x) unlist(x)
w <- art$weights
W <- list()
for (nm in names(w)) {
  v <- w[[nm]]
  # Matrices are JSON list-of-lists; vectors are flat lists.
  if (is.list(v[[1]])) W[[nm]] <- as_mat(v) else W[[nm]] <- as_vec(v)
}
s_mat <- as_mat(art$s_mat)

cat(sprintf("Compiling gmlp_decode.stan... "))
t0 <- Sys.time()
expose_stan_functions(stanc(file = here("gmlp_decode.stan")))
cat(sprintf("done (%.1fs)\n", as.numeric(Sys.time() - t0, units = "secs")))

tol <- 1e-4
max_diff <- 0
for (i in seq_along(art$test_cases)) {
  case <- art$test_cases[[i]]
  z    <- unlist(case$z)
  cond <- unlist(case$cond)
  expected <- unlist(case$out)
  got <- decode(
    z, cond, s_mat,
    W$embed_W0, W$embed_b0, W$embed_W1, W$embed_b1,
    W$ln0_scale, W$ln0_bias, W$ln1_scale, W$ln1_bias, W$ln2_scale, W$ln2_bias,
    W$sgu_norm_scale, W$sgu_norm_bias,
    W$b0_pi_W0, W$b0_pi_b0, W$b0_pi_W1, W$b0_pi_b1,
    W$b0_sgu_W, W$b0_sgu_b,
    W$b0_po_W0, W$b0_po_b0, W$b0_po_W1, W$b0_po_b1,
    W$b1_pi_W0, W$b1_pi_b0, W$b1_pi_W1, W$b1_pi_b1,
    W$b1_sgu_W, W$b1_sgu_b,
    W$b1_po_W0, W$b1_po_b0, W$b1_po_W1, W$b1_po_b1,
    W$head_W0, W$head_b0, W$head_W1, W$head_b1
  )
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
