# Active-*query* evaluation of the spline / interpolator: the companion to
# test-spline-ad.R (which differentiates w.r.t. knot *values*). Here the knot
# values are frozen (double) and the query point carries the active scalar, so
# the interpolated value differentiates w.r.t. where it is read -- the
# sensitivity a plant reading a frozen light profile at its own (active) height
# needs. Forward mode, so no tape / shared library is required.

compile_spline_query_interface <- function() {
  include_dir <- dirname(dirname(resolve_test_path(
    "include/odelia/spline.hpp", "inst/include/odelia/spline.hpp")))
  withr::local_envvar(PKG_CPPFLAGS = paste0("-I", shQuote(include_dir)))

  Rcpp::sourceCpp(code = '
    #include <Rcpp.h>
    #include <vector>
    #include <XAD/XAD.hpp>
    #include <odelia/spline.hpp>
    #include <odelia/interpolator.hpp>
    using fwd = xad::fwd<double>::active_type;

    // Value + dvalue/dquery at an active query, through the Interpolator wrapper
    // plant uses, with frozen (double) knot values. Returns the analytic slope
    // and the double-path value for comparison.
    // [[Rcpp::export]]
    Rcpp::NumericVector interp_query_deriv(std::vector<double> x,
                                           std::vector<double> y, double q) {
      odelia::interpolator::basic_interpolator<double> I;
      I.init(x, y);
      fwd qq(q); xad::derivative(qq) = 1.0;
      fwd v = I(qq);
      return Rcpp::NumericVector::create(
        Rcpp::_["value"]          = xad::value(v),
        Rcpp::_["dvalue_dquery"]  = xad::derivative(v),
        Rcpp::_["value_double"]   = I(q),
        Rcpp::_["deriv_analytic"] = I.deriv(q));
    }', verbose = FALSE)
}

testthat::test_that("active-query interpolation matches the double value and analytic slope", {
  compile_spline_query_interface()

  x <- as.numeric(0:6)
  y <- sin(x)                    # frozen knot values
  q <- 1.7

  r <- interp_query_deriv(x, y, q)
  # active-query value == double-path value (same polynomial, materialised in S)
  expect_equal(unname(r["value"]), unname(r["value_double"]))
  # dvalue/dquery == the spline's analytic first derivative
  expect_equal(unname(r["dvalue_dquery"]), unname(r["deriv_analytic"]), tolerance = 1e-12)
  # and consistent with a central finite difference of the value
  h <- 1e-6
  fd <- (interp_query_deriv(x, y, q + h)["value"] -
         interp_query_deriv(x, y, q - h)["value"]) / (2 * h)
  expect_equal(unname(r["dvalue_dquery"]), unname(fd), tolerance = 1e-6)
})
