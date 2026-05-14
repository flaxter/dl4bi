# Convert decoder JSONs emitted by train_decoders.py into canonical .rds
# files matching DESIGN.md §2.2. Reads manifest.json from input_dir and
# writes one .rds per decoder plus a copy of manifest.json to output_dir.
#
# Usage:
#   Rscript benchmarks/vae/rstan/pack_decoders.R <input_dir> [<output_dir>]
#
# If output_dir is omitted, .rds files are written next to the JSONs.

suppressPackageStartupMessages({
  library(jsonlite)
  library(digest)
})

SCHEMA_VERSION <- 1L
SUPPORTED_ARCH <- "MLPDeepRV"
SUPPORTED_DOMAIN <- "unit_interval"

# Stable sha256 fingerprint. Must mirror train_decoders.py:fingerprint().
# Floats are formatted as %.8f so the byte sequence is identical to what
# Python produces — no language-specific float repr.
deeprv_fingerprint <- function(arch, arch_version, domain, grid_size, kernel,
                               ls_lo, ls_hi) {
  canonical <- paste(arch, arch_version, domain, as.integer(grid_size), kernel,
                     sprintf("%.8f", as.numeric(ls_lo)),
                     sprintf("%.8f", as.numeric(ls_hi)),
                     sep = "|")
  digest::digest(canonical, algo = "sha256", serialize = FALSE)
}

# Convert one JSON artifact (R list, from fromJSON simplifyVector=FALSE) into
# the canonical .rds list shape.
build_rds_list <- function(j) {
  stopifnot(j$schema_version == SCHEMA_VERSION)
  stopifnot(j$arch == SUPPORTED_ARCH)
  stopifnot(j$domain == SUPPORTED_DOMAIN)
  L <- as.integer(j$L)
  stopifnot(as.integer(j$grid_size) == L)

  # JSON nested arrays come back as list-of-lists; flatten into matrices.
  to_matrix <- function(rows) {
    do.call(rbind, lapply(rows, function(r) as.numeric(unlist(r))))
  }
  W1 <- to_matrix(j$weights$W1)
  W2 <- to_matrix(j$weights$W2)
  b1 <- as.numeric(unlist(j$weights$b1))
  b2 <- as.numeric(unlist(j$weights$b2))

  # Sanity check shapes. W1 maps (L + cond_dim) -> hidden; W2 maps hidden -> L.
  # cond_dim = 1 for v0.1 (ls only).
  cond_dim <- length(j$conditionals)
  stopifnot(nrow(W1) == L + cond_dim)
  stopifnot(ncol(W1) == length(b1))
  stopifnot(nrow(W2) == length(b1))
  stopifnot(ncol(W2) == L && length(b2) == L)

  test_cases <- lapply(j$test_cases, function(tc) {
    list(
      z    = as.numeric(unlist(tc$z)),
      cond = as.numeric(unlist(tc$cond)),
      out  = as.numeric(unlist(tc$out))
    )
  })

  list(
    schema_version   = SCHEMA_VERSION,
    arch             = SUPPORTED_ARCH,
    arch_version     = as.character(j$arch_version),
    domain           = SUPPORTED_DOMAIN,
    grid_size        = L,
    L                = L,
    grid_coords      = as.numeric(unlist(j$grid_coords)),
    kernel           = as.character(j$kernel),
    conditionals     = as.character(unlist(j$conditionals)),
    ls_trained_range = as.numeric(unlist(j$ls_trained_range)),
    weights = list(W1 = W1, b1 = b1, W2 = W2, b2 = b2),
    training = list(
      steps            = as.integer(j$training$steps),
      batch            = as.integer(j$training$batch),
      lr               = as.numeric(j$training$lr),
      optimizer        = as.character(j$training$optimizer),
      weight_decay     = as.numeric(j$training$weight_decay),
      clip_global_norm = as.numeric(j$training$clip_global_norm),
      jitter           = as.numeric(j$training$jitter),
      final_mse        = as.numeric(j$training$final_mse),
      seed             = as.integer(j$training$seed),
      wall_time_sec    = as.numeric(j$training$wall_time_sec)
    ),
    test_cases  = test_cases,
    fingerprint = as.character(j$fingerprint)
  )
}

main <- function(args) {
  if (length(args) < 1) {
    stop("usage: Rscript pack_decoders.R <input_dir> [<output_dir>]")
  }
  input_dir  <- args[1]
  output_dir <- if (length(args) >= 2) args[2] else input_dir
  stopifnot(dir.exists(input_dir))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  manifest_in <- file.path(input_dir, "manifest.json")
  if (!file.exists(manifest_in)) {
    stop(sprintf("missing manifest.json in %s — run train_decoders.py first",
                 input_dir))
  }
  manifest <- fromJSON(manifest_in, simplifyVector = FALSE)
  stopifnot(manifest$schema_version == SCHEMA_VERSION)
  stopifnot(manifest$arch == SUPPORTED_ARCH)

  out_entries <- list()
  for (fp in names(manifest$decoders)) {
    json_name <- manifest$decoders[[fp]]
    json_path <- file.path(input_dir, json_name)
    if (!file.exists(json_path)) {
      stop(sprintf("manifest references missing file: %s", json_path))
    }
    j <- fromJSON(json_path, simplifyVector = FALSE)
    if (as.character(j$fingerprint) != fp) {
      stop(sprintf("manifest fingerprint %s != JSON fingerprint %s for %s",
                   fp, j$fingerprint, json_name))
    }

    # Verify fingerprint matches what R computes from the source fields.
    # If this diverges, train_decoders.py and pack_decoders.R drift apart.
    computed <- deeprv_fingerprint(
      j$arch, j$arch_version, j$domain, j$grid_size, j$kernel,
      j$ls_trained_range[[1]], j$ls_trained_range[[2]]
    )
    if (computed != fp) {
      stop(sprintf(paste0("fingerprint divergence for %s: ",
                          "stored=%s computed=%s. Python and R fingerprint ",
                          "functions have drifted — sync them before shipping."),
                   json_name, fp, computed))
    }

    rds <- build_rds_list(j)
    rds_name <- sub("\\.json$", ".rds", json_name)
    rds_path <- file.path(output_dir, rds_name)
    saveRDS(rds, rds_path, version = 2)
    cat(sprintf("  %s -> %s (%s bytes)\n", json_name, rds_name,
                format(file.info(rds_path)$size, big.mark = ",")))
    out_entries[[fp]] <- rds_name
  }

  out_manifest <- list(
    schema_version   = SCHEMA_VERSION,
    arch             = manifest$arch,
    arch_version     = manifest$arch_version,
    domain           = manifest$domain,
    ls_trained_range = manifest$ls_trained_range,
    decoders         = out_entries
  )
  manifest_out <- file.path(output_dir, "manifest.json")
  write(toJSON(out_manifest, auto_unbox = TRUE, pretty = TRUE), manifest_out)
  cat(sprintf("\nWrote %d .rds files + %s\n",
              length(out_entries), manifest_out))
}

main(commandArgs(trailingOnly = TRUE))
