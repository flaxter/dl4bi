# DESIGN.md §4.9: "the forward-pass match is the most important test."
# Skipped when rstan isn't installed so unit tests can still run in
# minimal dev envs.

skip_if_no_rstan <- function() {
  skip_if_not_installed("rstan")
}

stan_decode_fn <- (function() {
  cached <- NULL
  function() {
    if (!is.null(cached)) return(cached)
    skip_if_no_rstan()
    stan_file <- system.file("stan", "decode_mlp.stan",
                             package = "brms.deeprv")
    expect_true(nzchar(stan_file) && file.exists(stan_file))
    # Compile + expose. Slow on first run (1-2 min); cached for the rest
    # of the test session via the closure.
    fn_env <- new.env()
    rstan::expose_stan_functions(rstan::stanc(file = stan_file),
                                 env = fn_env, show_compiler_warnings = FALSE)
    cached <<- fn_env$decode
    cached
  }
})()

forward_match_one <- function(dr) {
  decode <- stan_decode_fn()
  W1 <- dr$weights$W1; b1 <- dr$weights$b1
  W2 <- dr$weights$W2; b2 <- dr$weights$b2
  expect_true(length(dr$test_cases) >= 1L,
              info = sprintf("%s: no test cases", dr$fingerprint))
  for (i in seq_along(dr$test_cases)) {
    tc <- dr$test_cases[[i]]
    got <- decode(tc$z, tc$cond, W1, b1, W2, b2)
    expect_equal(as.numeric(got), as.numeric(tc$out), tolerance = 1e-5,
                 info = sprintf("decoder %s case %d", dr$fingerprint, i - 1))
  }
}

test_that("every shipped decoder reproduces its JAX test cases (< 1e-5)", {
  skip_if_no_rstan()
  m <- jsonlite::fromJSON(
    system.file("extdata", "decoders", "manifest.json",
                package = "brms.deeprv"),
    simplifyVector = FALSE
  )
  # Pull every (domain, grid_size, kernel) triple advertised by the
  # manifest, loop, and forward-match each one.
  # Filenames have the canonical form <domain>_<grid_size>_<kernel>.rds
  # where <domain> can itself contain underscores ("unit_interval"), so
  # peel a known kernel suffix from the end rather than splitting blindly.
  candidates <- c("matern_1_2", "matern_3_2", "matern_5_2", "rbf")
  for (fp in names(m$decoders)) {
    stem <- sub("\\.rds$", "", m$decoders[[fp]])
    kernel <- NA_character_
    for (k in candidates) {
      suf <- paste0("_", k)
      if (endsWith(stem, suf)) {
        kernel <- k
        rest <- substr(stem, 1, nchar(stem) - nchar(suf))
        break
      }
    }
    expect_false(is.na(kernel),
                 info = sprintf("can't parse kernel from %s", stem))
    rest_parts <- strsplit(rest, "_")[[1]]
    grid_size <- suppressWarnings(as.integer(tail(rest_parts, 1)))
    domain <- paste(rest_parts[-length(rest_parts)], collapse = "_")
    expect_false(is.na(grid_size),
                 info = sprintf("can't parse grid_size from %s", stem))

    dr <- load_deeprv(domain, grid_size = grid_size, kernel = kernel)
    forward_match_one(dr)
  }
})
