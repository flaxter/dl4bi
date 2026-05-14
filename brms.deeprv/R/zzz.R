.onLoad <- function(libname, pkgname) {
  # Allow tests / power users to override the bundled catalog without
  # reinstalling the package. The override sets where decoder_dir() and
  # manifest_path() look. Useful during package development before the
  # full v0.1 catalog is shipped — point this at the smoke artifacts.
  override <- Sys.getenv("BRMS_DEEPRV_DECODER_DIR", unset = "")
  if (nzchar(override)) {
    options(brms.deeprv.decoder_dir = override)
  }
}
