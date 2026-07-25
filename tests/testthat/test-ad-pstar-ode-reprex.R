# Fast-iteration harness + regression witness for the nested interior-optimum
# composition: an implicit_value p* node whose residual is a central difference of
# a profit that itself contains nested implicit_value nodes, evaluated inside
# ode_rates and replayed through compute_gradient.
#
# Built to bisect plant's TF24 interior-p* reverse-sweep segfault without a ~6
# minute plant rebuild. It does NOT currently reproduce that crash (see the file
# header of inst/examples/pstar_ode_reprex_interface.cpp for what that rules out);
# it stands as the witness that this composition is correct in isolation, and as
# the harness to extend when adding the remaining plant-specific axes.

ensure_pstar_ode_reprex <- function() {
  skip_on_cran()
  dlls <- getLoadedDLLs()
  skip_if_not("odelia" %in% names(dlls), "odelia DLL not loaded")
  odelia_so <- dlls[["odelia"]][["path"]]
  odelia_inc <- system.file("include", package = "odelia")
  src <- system.file("examples", "pstar_ode_reprex_interface.cpp", package = "odelia")
  skip_if(odelia_inc == "" || src == "", "odelia headers/examples unavailable")
  skip_if(!file.exists(odelia_so), "compiled odelia library unavailable")

  old <- Sys.getenv(c("PKG_CXXFLAGS", "PKG_LIBS"), unset = NA)
  Sys.setenv(PKG_CXXFLAGS = paste0("-I", odelia_inc))
  Sys.setenv(PKG_LIBS = shQuote(odelia_so))
  on.exit({
    for (k in names(old)) {
      if (is.na(old[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv, setNames(list(old[[k]]), k))
    }
  }, add = TRUE)

  built <- tryCatch({
    Rcpp::sourceCpp(src)
    TRUE
  }, error = function(e) {
    message("pstar_ode_reprex sourceCpp build failed: ", conditionMessage(e))
    FALSE
  })
  skip_if_not(built, "sourceCpp build unavailable in this session")
  invisible(TRUE)
}

test_that("frozen p* reproduces the double value and FD-matches exactly", {
  ensure_pstar_ode_reprex()
  r <- pstar_ode_reprex(use_pstar = 0L, nest = 1L, nlayer = 2L, n_steps = 20L)
  # The active run must reproduce the double trajectory bit-for-bit.
  expect_equal(r$value, r$value_double, tolerance = 1e-12)
  # With no argmax node in the chain the adjoint is exact, not FD-approximate.
  expect_equal(r$grad_kmax, r$grad_kmax_fd, tolerance = 1e-6)
  expect_equal(r$grad_c, r$grad_c_fd, tolerance = 1e-6)
})

test_that("nested interior-optimum p* node inside ode_rates gives a sane gradient", {
  ensure_pstar_ode_reprex()
  r <- pstar_ode_reprex(use_pstar = 1L, nest = 1L, nlayer = 2L, n_steps = 20L)
  expect_equal(r$value, r$value_double, tolerance = 1e-12)
  # p* is defined by a central-difference stationarity residual (eps = 1e-2), so
  # the gradient carries that quadrature error rather than matching FD to 1e-9.
  expect_equal(r$grad_kmax, r$grad_kmax_fd, tolerance = 0.05)
  expect_true(is.finite(r$grad_c))
})

test_that("the composition survives scale, persistent members and unseeded fields", {
  ensure_pstar_ode_reprex()
  # Each argument here is one of the axes suspected of triggering plant's crash;
  # all on at once must still complete and FD-match.
  r <- pstar_ode_reprex(use_pstar = 1L, nest = 1L, nlayer = 5L, n_steps = 100L,
                        persist = 1L, nfields = 40L)
  expect_equal(r$value, r$value_double, tolerance = 1e-12)
  expect_equal(r$grad_kmax, r$grad_kmax_fd, tolerance = 0.05)
})
