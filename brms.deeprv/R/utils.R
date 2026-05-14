#' SHA-256 fingerprint of a decoder identity.
#'
#' Must produce byte-identical output to `fingerprint()` in
#' `benchmarks/vae/rstan/train_decoders.py` and to `deeprv_fingerprint()`
#' in `benchmarks/vae/rstan/pack_decoders.R`. Floats are formatted with
#' `%.8f` - see DESIGN.md section 4.8 for why nothing else round-trips between
#' Python and R.
#'
#' @keywords internal
#' @noRd
deeprv_fingerprint <- function(arch, arch_version, domain, grid_size, kernel,
                               ls_lo, ls_hi) {
  canonical <- paste(
    arch, arch_version, domain, as.integer(grid_size), kernel,
    sprintf("%.8f", as.numeric(ls_lo)),
    sprintf("%.8f", as.numeric(ls_hi)),
    sep = "|"
  )
  digest::digest(canonical, algo = "sha256", serialize = FALSE)
}

# Schema constants. Bumping SCHEMA_VERSION or SUPPORTED_ARCH_VERSION
# invalidates existing .rds artifacts; load_deeprv() errors with a
# pointer to retrain.
SCHEMA_VERSION <- 1L
SUPPORTED_ARCHES <- c("MLPDeepRV")
SUPPORTED_DOMAINS <- c("unit_interval", "unit_square")
SUPPORTED_KERNELS <- c("matern_1_2", "matern_3_2", "matern_5_2", "rbf")

#' Validate that a loaded decoder list has the canonical structure of
#' DESIGN.md section 2.2 and that its fingerprint matches what we'd recompute
#' from the source fields. Errors loudly on any mismatch.
#'
#' @keywords internal
#' @noRd
validate_decoder <- function(dr, source = "<unknown>") {
  if (!is.list(dr)) {
    stop(sprintf("decoder %s is not a list", source), call. = FALSE)
  }
  required <- c("schema_version", "arch", "arch_version", "domain",
                "grid_size", "L", "grid_coords", "kernel", "conditionals",
                "ls_trained_range", "weights", "fingerprint")
  missing <- setdiff(required, names(dr))
  if (length(missing) > 0) {
    stop(sprintf("decoder %s missing fields: %s",
                 source, paste(missing, collapse = ", ")),
         call. = FALSE)
  }
  if (!identical(as.integer(dr$schema_version), SCHEMA_VERSION)) {
    stop(sprintf("decoder %s has schema_version=%s, package expects %d. ",
                 source, dr$schema_version, SCHEMA_VERSION),
         "Retrain the catalog or install a matching package version.",
         call. = FALSE)
  }
  if (!(dr$arch %in% SUPPORTED_ARCHES)) {
    stop(sprintf("decoder %s has arch=%s, supported: %s",
                 source, dr$arch, paste(SUPPORTED_ARCHES, collapse = ", ")),
         call. = FALSE)
  }
  if (!(dr$domain %in% SUPPORTED_DOMAINS)) {
    stop(sprintf("decoder %s has domain=%s, supported: %s",
                 source, dr$domain, paste(SUPPORTED_DOMAINS, collapse = ", ")),
         call. = FALSE)
  }
  if (!(dr$kernel %in% SUPPORTED_KERNELS)) {
    stop(sprintf("decoder %s has kernel=%s, supported: %s",
                 source, dr$kernel, paste(SUPPORTED_KERNELS, collapse = ", ")),
         call. = FALSE)
  }
  if (!identical(as.integer(dr$L), as.integer(dr$grid_size))) {
    stop(sprintf("decoder %s: L=%s != grid_size=%s",
                 source, dr$L, dr$grid_size),
         call. = FALSE)
  }
  if (length(dr$grid_coords) != as.integer(dr$L)) {
    stop(sprintf("decoder %s: grid_coords length=%d, L=%d",
                 source, length(dr$grid_coords), as.integer(dr$L)),
         call. = FALSE)
  }
  if (length(dr$ls_trained_range) != 2L ||
      !is.numeric(dr$ls_trained_range) ||
      dr$ls_trained_range[1] >= dr$ls_trained_range[2]) {
    stop(sprintf("decoder %s: ls_trained_range must be a length-2 numeric ",
                 source),
         "with lo < hi.", call. = FALSE)
  }
  w <- dr$weights
  if (!is.list(w) || !all(c("W1", "b1", "W2", "b2") %in% names(w))) {
    stop(sprintf("decoder %s: weights must be a list(W1, b1, W2, b2)",
                 source),
         call. = FALSE)
  }
  L <- as.integer(dr$L)
  C <- length(dr$conditionals)
  if (!is.matrix(w$W1) || nrow(w$W1) != L + C) {
    stop(sprintf("decoder %s: W1 must be a matrix with %d rows (L+cond_dim), ",
                 source, L + C),
         sprintf("got %s", paste(dim(w$W1), collapse = "x")),
         call. = FALSE)
  }
  hidden <- ncol(w$W1)
  if (length(w$b1) != hidden) {
    stop(sprintf("decoder %s: b1 length=%d != hidden=%d",
                 source, length(w$b1), hidden),
         call. = FALSE)
  }
  if (!is.matrix(w$W2) || nrow(w$W2) != hidden || ncol(w$W2) != L) {
    stop(sprintf("decoder %s: W2 must be (%d, %d), got %s",
                 source, hidden, L, paste(dim(w$W2), collapse = "x")),
         call. = FALSE)
  }
  if (length(w$b2) != L) {
    stop(sprintf("decoder %s: b2 length=%d != L=%d",
                 source, length(w$b2), L),
         call. = FALSE)
  }
  computed <- deeprv_fingerprint(
    dr$arch, dr$arch_version, dr$domain, dr$grid_size, dr$kernel,
    dr$ls_trained_range[1], dr$ls_trained_range[2]
  )
  if (!identical(computed, as.character(dr$fingerprint))) {
    stop(sprintf("decoder %s: fingerprint mismatch. ", source),
         sprintf("stored=%s, recomputed=%s. ",
                 dr$fingerprint, computed),
         "The .rds was probably built with an out-of-date pack_decoders.R ",
         "or a different float-formatting convention. See DESIGN.md section 4.8.",
         call. = FALSE)
  }
  invisible(TRUE)
}
