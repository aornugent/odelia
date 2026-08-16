# Tests for forward_derivative (the one-input forward-mode derivative).
# Forward mode records nothing, so this needs only the include path -- none of the
# tape instantiations in the odelia shared library are reached.

compile_forward_derivative_interface <- function() {
  include_dir <- odelia_include_dir()
  withr::local_envvar(PKG_CPPFLAGS = odelia_cppflags(include_dir))
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <cmath>
    #include <odelia/gradient.hpp>

    // [[Rcpp::export]]
    double fd_cubic(double x, double a, double b) {
      return odelia::ode::forward_derivative(x, [&](auto v) -> decltype(v) {
        return a * v * v * v + b * v;
      });
    }

    // [[Rcpp::export]]
    double fd_saturating(double x, double scale, double b, double c) {
      return odelia::ode::forward_derivative(x, [&](auto v) -> decltype(v) {
        return scale * (1.0 - exp(-pow(v / b, c)));
      });
    }

    // [[Rcpp::export]]
    double fd_colimited(double x, double p, double q, double curv) {
      return odelia::ode::forward_derivative(x, [&](auto v) -> decltype(v) {
        decltype(v) r = p * v;
        decltype(v) e = q * v / (v + 1.0);
        decltype(v) s = r + e;
        return (s - sqrt(s * s - 4.0 * curv * r * e)) / (2.0 * curv);
      });
    }', verbose = FALSE)
}

testthat::test_that("forward_derivative is exact on a polynomial", {
  compile_forward_derivative_interface()

  # d/dx (a x^3 + b x) = 3 a x^2 + b, which is representable exactly here, so the
  # forward sweep should return it bit for bit rather than to a tolerance.
  a <- 2.5
  b <- -0.75
  for (x in c(-3.25, -0.5, 0.125, 1.75, 4.0)) {
    testthat::expect_identical(fd_cubic(x, a, b), 3 * a * x^2 + b)
  }
})

testthat::test_that("forward_derivative differentiates a pow/exp composite", {
  compile_forward_derivative_interface()

  # scale * (1 - exp(-(x/b)^c)); the analytic derivative is
  # scale * exp(-(x/b)^c) * c/b * (x/b)^(c-1).
  scale <- 46.32995
  b <- 3
  cc <- 2.04
  x <- c(0.4, 1.1, 2.6, 5.3)
  analytic <- scale * exp(-(x / b)^cc) * (cc / b) * (x / b)^(cc - 1)
  got <- vapply(x, fd_saturating, numeric(1), scale = scale, b = b, c = cc)

  testthat::expect_equal(got, analytic, tolerance = 1e-12)
})

testthat::test_that("forward_derivative agrees with a central difference of the value", {
  compile_forward_derivative_interface()

  # A colimitation minimum, whose sqrt makes the derivative worth checking against
  # the function it is meant to be the derivative of.
  p <- 1.4
  q <- 9.0
  curv <- 0.99
  value <- function(v) {
    r <- p * v
    e <- q * v / (v + 1)
    s <- r + e
    (s - sqrt(s^2 - 4 * curv * r * e)) / (2 * curv)
  }
  x <- c(0.6, 1.3, 3.7, 8.2)
  h <- 1e-6
  fd <- (value(x + h) - value(x - h)) / (2 * h)
  got <- vapply(x, fd_colimited, numeric(1), p = p, q = q, curv = curv)

  testthat::expect_equal(got, fd, tolerance = 1e-7)
})
