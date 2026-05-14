# Path to the decoder catalog. Defaults to inst/extdata/decoders/ inside
# the installed package; can be overridden via `options(brms.deeprv.decoder_dir)`
# or the `BRMS_DEEPRV_DECODER_DIR` env var (resolved in .onLoad()). The
# override is necessary during package development because the full v0.1
# catalog (~tens of MB of .rds) is not bundled with the source tree.
decoder_dir <- function() {
  override <- getOption("brms.deeprv.decoder_dir", "")
  if (nzchar(override)) {
    return(override)
  }
  system.file("extdata", "decoders", package = "brms.deeprv")
}

manifest_path <- function() {
  file.path(decoder_dir(), "manifest.json")
}

# Cache the manifest so repeated load_deeprv() calls don't hit disk.
.deeprv_env <- new.env(parent = emptyenv())

read_manifest <- function(path = manifest_path()) {
  if (!is.null(.deeprv_env$manifest) &&
      identical(.deeprv_env$manifest_path, path)) {
    return(.deeprv_env$manifest)
  }
  if (!nzchar(path) || !file.exists(path)) {
    stop("brms.deeprv: no decoder manifest found. The package was ",
         "installed without a bundled catalog. Re-install with the ",
         "decoders/ directory populated, or point to an external ",
         "catalog with options(brms.deeprv.decoder_dir = ...).",
         call. = FALSE)
  }
  m <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(as.integer(m$schema_version), SCHEMA_VERSION)) {
    stop(sprintf("manifest schema_version=%s, package expects %d",
                 m$schema_version, SCHEMA_VERSION),
         call. = FALSE)
  }
  if (!is.list(m$decoders) || length(m$decoders) == 0) {
    stop("manifest has no decoders", call. = FALSE)
  }
  .deeprv_env$manifest <- m
  .deeprv_env$manifest_path <- path
  m
}

#' Load a pre-trained DeepRV decoder
#'
#' Looks up a decoder by `(domain, grid_size, kernel)` against the
#' shipped catalog manifest, validates its fingerprint, and returns a
#' `deepRV_decoder` object. Use the returned object as the `decoder`
#' argument of [deepRV()] (forthcoming).
#'
#' Decoder weights are MLP outputs of `MLPDeepRV(dims = [L, L])` trained
#' to emulate `chol(K_ls) %*% z` for length-scale `ls` drawn from the
#' decoder's `ls_trained_range`. Inference outside that range is not
#' supported - see DESIGN.md section 5.
#'
#' @param domain Coordinate domain - `"unit_interval"` is the only
#'   value supported as a 1D decoder in v0.1. Use [load_deeprv_kron()]
#'   for `unit_square`.
#' @param grid_size Number of grid points. Must equal one of the sizes
#'   in the shipped catalog (see [list_decoders()]).
#' @param kernel Covariance kernel: `"matern_1_2"`, `"matern_3_2"`,
#'   `"matern_5_2"`, or `"rbf"`.
#'
#' @return A list with class `"deepRV_decoder"` and the schema of
#'   DESIGN.md section 2.2. The relevant accessors for downstream use are
#'   `$grid_coords` (the exact training grid), `$ls_trained_range`, and
#'   `$weights`.
#'
#' @examples
#' \dontrun{
#' dr <- load_deeprv("unit_interval", grid_size = 100, kernel = "matern_3_2")
#' print(dr)
#' }
#' @export
load_deeprv <- function(domain, grid_size, kernel) {
  if (!is.character(domain) || length(domain) != 1L) {
    stop("`domain` must be a single string", call. = FALSE)
  }
  if (domain == "unit_square") {
    stop("Use load_deeprv_kron() for unit_square decoders. ",
         "load_deeprv() returns 1D decoders only.",
         call. = FALSE)
  }
  if (!(domain %in% SUPPORTED_DOMAINS)) {
    stop(sprintf("domain=%s not supported. supported: %s",
                 domain, paste(SUPPORTED_DOMAINS, collapse = ", ")),
         call. = FALSE)
  }
  if (!is.numeric(grid_size) || length(grid_size) != 1L) {
    stop("`grid_size` must be a single number", call. = FALSE)
  }
  if (!(kernel %in% SUPPORTED_KERNELS)) {
    stop(sprintf("kernel=%s not supported. supported: %s",
                 kernel, paste(SUPPORTED_KERNELS, collapse = ", ")),
         call. = FALSE)
  }

  m <- read_manifest()
  if (!identical(m$domain, domain)) {
    stop(sprintf("manifest domain=%s, requested domain=%s",
                 m$domain, domain),
         call. = FALSE)
  }
  fp <- deeprv_fingerprint(
    m$arch, m$arch_version, domain, grid_size, kernel,
    m$ls_trained_range[[1]], m$ls_trained_range[[2]]
  )
  rds_name <- m$decoders[[fp]]
  if (is.null(rds_name)) {
    avail <- describe_manifest(m)
    stop(sprintf(
      "no decoder for (domain=%s, grid_size=%s, kernel=%s) in the shipped catalog. ",
      domain, grid_size, kernel),
      "available:\n", paste(avail, collapse = "\n"),
      call. = FALSE)
  }
  rds_path <- file.path(dirname(manifest_path()), rds_name)
  if (!file.exists(rds_path)) {
    stop(sprintf("manifest points to %s but the file is missing", rds_path),
         call. = FALSE)
  }
  dr <- readRDS(rds_path)
  validate_decoder(dr, source = rds_name)
  class(dr) <- c("deepRV_decoder", "list")
  dr
}

# Human-readable listing of what's in a manifest, for "decoder not found" errors.
describe_manifest <- function(m) {
  lines <- character(0)
  for (fp in names(m$decoders)) {
    fname <- m$decoders[[fp]]
    # Filenames have the canonical form <domain>_<grid_size>_<kernel>.rds -
    # parsing this avoids loading every .rds just to list options.
    stem <- sub("\\.rds$", "", fname)
    lines <- c(lines, sprintf("  %s  (%s)", stem, substr(fp, 1, 12)))
  }
  sort(lines)
}

#' Load a Kronecker-composed 2D decoder for `unit_square`
#'
#' Pairs two 1D `unit_interval` decoders (one along x, one along y).
#' Sampling reduces to `F = L_x %*% Z %*% t(L_y)` at fit time. Only
#' axis-wise separable kernels are valid here - Matern / RBF are
#' interpreted as separable in v0.1, not as isotropic 2D kernels. See
#' DESIGN.md section 2.5.
#'
#' @param domain Must be `"unit_square"`.
#' @param grid_side Number of points per axis; the full grid has
#'   `grid_side^2` cells.
#' @param kernel One of `"matern_1_2"`, `"matern_3_2"`, `"matern_5_2"`,
#'   `"rbf"` - applied identically along both axes.
#'
#' @return A list with class `c("deepRV_decoder_kron", "deepRV_decoder")`
#'   and the schema of DESIGN.md section 2.5. `$x_decoder` and `$y_decoder`
#'   hold the 1D decoders; `$grid_coords` is the full 2D grid as a
#'   `grid_side^2 x 2` matrix.
#'
#' @export
load_deeprv_kron <- function(domain, grid_side, kernel) {
  if (!identical(domain, "unit_square")) {
    stop("load_deeprv_kron() requires domain=\"unit_square\"", call. = FALSE)
  }
  if (!is.numeric(grid_side) || length(grid_side) != 1L) {
    stop("`grid_side` must be a single number", call. = FALSE)
  }
  if (!(kernel %in% SUPPORTED_KERNELS)) {
    stop(sprintf("kernel=%s not supported. supported: %s",
                 kernel, paste(SUPPORTED_KERNELS, collapse = ", ")),
         call. = FALSE)
  }
  # Both axes share the same 1D decoder - DESIGN.md section 2.5 requires that
  # the kernel is identical along x and y in v0.1.
  axis_dec <- load_deeprv("unit_interval", grid_size = grid_side,
                          kernel = kernel)
  N <- as.integer(axis_dec$L)
  # Full grid as (N^2 x 2): col 1 varies fastest = x, then y. This
  # matches the Stan-side flattening F[i, j] -> idx = (j - 1) * N + i.
  coords1d <- axis_dec$grid_coords
  full_grid <- cbind(
    rep(coords1d, times = N),
    rep(coords1d, each = N)
  )
  out <- list(
    schema_version    = SCHEMA_VERSION,
    arch              = "MLPDeepRV",
    arch_version      = axis_dec$arch_version,
    domain            = "unit_square",
    grid_side         = N,
    L                 = N * N,
    grid_coords       = full_grid,
    kernel            = kernel,
    x_decoder         = axis_dec,
    y_decoder         = axis_dec,
    conditionals      = c("ls_x", "ls_y"),
    ls_trained_range  = axis_dec$ls_trained_range
  )
  class(out) <- c("deepRV_decoder_kron", "deepRV_decoder", "list")
  out
}

#' @export
print.deepRV_decoder <- function(x, ...) {
  cat("<deepRV_decoder>\n")
  cat(sprintf("  arch        : %s (v%s)\n", x$arch, x$arch_version))
  cat(sprintf("  domain      : %s\n", x$domain))
  cat(sprintf("  grid_size   : %d  (L = %d)\n", x$grid_size, x$L))
  cat(sprintf("  kernel      : %s\n", x$kernel))
  cat(sprintf("  ls range    : [%g, %g]\n",
              x$ls_trained_range[1], x$ls_trained_range[2]))
  cat(sprintf("  fingerprint : %s\n", substr(x$fingerprint, 1, 16)))
  if (!is.null(x$training)) {
    cat(sprintf("  training    : %s steps, MSE=%.4g\n",
                format(x$training$steps, big.mark = ","),
                x$training$final_mse))
  }
  invisible(x)
}

#' @export
print.deepRV_decoder_kron <- function(x, ...) {
  cat("<deepRV_decoder_kron>\n")
  cat(sprintf("  domain      : %s\n", x$domain))
  cat(sprintf("  grid_side   : %d  (L = %d)\n", x$grid_side, x$L))
  cat(sprintf("  kernel      : %s  (axis-wise separable)\n", x$kernel))
  cat(sprintf("  ls range    : [%g, %g]\n",
              x$ls_trained_range[1], x$ls_trained_range[2]))
  cat(sprintf("  x_decoder fp: %s\n",
              substr(x$x_decoder$fingerprint, 1, 16)))
  invisible(x)
}
