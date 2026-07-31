# Tests for odelia::ode::vector_jacobian_product: one block recorded and swept
# once, writing input adjoints into a caller-owned buffer.
#
# The XAD adjoint Tape<T,N> methods are instantiated only in src/Tape.cpp (in the
# odelia shared library), so this sourceCpp build links against it via PKG_LIBS,
# exactly like the leaf-thermal AD interface (see helper-load-odelia.R).

compile_vjp_interface <- function() {
  ensure_ode_interface_loaded(rebuild = FALSE)
  include_dir <- dirname(dirname(resolve_test_path(
    "include/odelia/ode_solver.hpp", "inst/include/odelia/ode_solver.hpp")))
  odelia_so <- .odelia_test_cache$odelia_so
  pkg_libs <- if (is.character(odelia_so) && length(odelia_so) == 1 &&
                  !is.na(odelia_so) && nzchar(odelia_so) && file.exists(odelia_so)) {
    shQuote(normalizePath(odelia_so, winslash = "/", mustWork = FALSE))
  } else {
    Sys.getenv("PKG_LIBS", unset = "")
  }
  withr::local_envvar(PKG_CPPFLAGS = paste0("-I", shQuote(include_dir)),
                      PKG_LIBS = pkg_libs)

  res <- tryCatch({
    Rcpp::sourceCpp(code = '
      #include <Rcpp.h>
      #include <cstddef>
      #include <vector>
      #include <XAD/XAD.hpp>
      #include <odelia/gradient.hpp>

      // A four-input, three-output block, small enough to difference centrally.
      template <class S>
      static void small_block(const std::vector<S>& x, std::vector<S>& y) {
        y[0] = sin(x[0]) * x[1] + x[2] * x[2] * x[3];
        y[1] = exp(0.5 * x[1]) - x[0] * x[3];
        y[2] = x[0] * x[1] * x[2] + log(1.0 + x[3] * x[3]);
      }

      // Fixed arithmetic on the first three entries; the rest are registered inputs
      // the block never reads, so the recording does not grow with their number.
      template <class S>
      static void padded_block(const std::vector<S>& x, std::vector<S>& y) {
        for (std::size_t i = 0; i < y.size(); ++i) {
          S v = sin(x[0]) * x[1] + x[2] * x[2] + S(double(i)) * x[1];
          y[i] = v;
        }
      }

      // [[Rcpp::export]]
      Rcpp::NumericVector small_block_value(std::vector<double> x) {
        std::vector<double> y(3);
        small_block(x, y);
        return Rcpp::wrap(y);
      }

      // [[Rcpp::export]]
      Rcpp::NumericVector small_block_vjp(std::vector<double> x,
                                          std::vector<double> output_adjoints) {
        std::vector<double> input_adjoints;
        odelia::ode::vector_jacobian_product(
            x, output_adjoints,
            [](const auto& xa, auto& ya) { small_block(xa, ya); }, input_adjoints);
        return Rcpp::wrap(input_adjoints);
      }

      // [[Rcpp::export]]
      double padded_block_recording_size(int n_inputs,
                                         std::vector<double> output_adjoints) {
        std::vector<double> x(n_inputs, 0.5), input_adjoints;
        return double(odelia::ode::vector_jacobian_product(
            x, output_adjoints,
            [](const auto& xa, auto& ya) { padded_block(xa, ya); }, input_adjoints));
      }

      // [[Rcpp::export]]
      Rcpp::NumericVector small_block_vjp_under_active_tape(
          std::vector<double> x, std::vector<double> output_adjoints) {
        xad::adj<double>::tape_type tape;
        return small_block_vjp(x, output_adjoints);
      }', verbose = FALSE)
    NULL
  }, error = function(e) e)

  if (inherits(res, "error")) {
    msg <- conditionMessage(res)
    if (grepl("active_tape_", msg, fixed = TRUE)) {
      testthat::skip("vector_jacobian_product sourceCpp symbols are unavailable in this load_all session; run installed-package tests for this context.")
    }
    stop(res)
  }
  invisible(TRUE)
}

testthat::test_that("vector_jacobian_product reproduces a central finite difference", {
  compile_vjp_interface()

  x <- c(0.7, -1.3, 0.45, 1.1)
  w <- c(1.0, -2.0, 0.5)

  # The product is t(J) %*% w, so its j-th entry is the derivative of w . f(x)
  # along input j -- which a central difference of the same block gives directly.
  h  <- 1e-5
  fd <- vapply(seq_along(x), function(j) {
    xp <- x; xp[j] <- xp[j] + h
    xm <- x; xm[j] <- xm[j] - h
    sum(w * (small_block_value(xp) - small_block_value(xm))) / (2 * h)
  }, numeric(1))

  expect_equal(small_block_vjp(x, w), fd, tolerance = 1e-8)
})

testthat::test_that("the recording size is a property of the block", {
  compile_vjp_interface()

  # One recording serves every output adjoint, so sweeping the same three-output
  # block with one seeded adjoint and with three gives the same recording size.
  # A product implemented as a loop of gradients would record once per output.
  expect_equal(padded_block_recording_size(10, c(1, 0, 0)),
               padded_block_recording_size(10, c(1, 2, 3)))

  # Inputs the block does not read add registered slots, not recorded operations,
  # so a tenfold input count must not multiply the recording size tenfold.
  small <- padded_block_recording_size(10, c(1, 2, 3))
  large <- padded_block_recording_size(100, c(1, 2, 3))
  expect_lt(large, 10 * small)
})

testthat::test_that("vector_jacobian_product stops when a tape is already active", {
  compile_vjp_interface()

  expect_error(small_block_vjp_under_active_tape(c(0.7, -1.3, 0.45, 1.1),
                                                 c(1.0, -2.0, 0.5)),
               "tape is already active")
})
