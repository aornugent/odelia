/* Exercises odelia::incomplete_gamma, compiled on demand by
 * test-ad-incomplete-gamma.R (sourceCpp).
 *
 * Returns the value (checked in R against pgamma(x,a)*gamma(a)), its d/dx (the
 * integrand x^{a-1} e^{-x}) and d/da (vs a finite difference), and the Weibull
 * antiderivative G(m) = (b/c) gamma(1/c, (m/b)^c) with dG/dm -- which must equal
 * the integrand exp(-(m/b)^c) exactly by the Leibniz endpoint -- and dG/dc (vs a
 * finite difference, the differentiated-trait channel through d/da).
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/incomplete_gamma.hpp>

#include <cmath>

using namespace Rcpp;

// Weibull antiderivative in plain double, for the finite-difference references.
static double weibull_int(double m, double b, double c) {
  return (b / c) * odelia::incomplete_gamma(1.0 / c, std::pow(m / b, c));
}

// [[Rcpp::export]]
Rcpp::List incomplete_gamma_demo(double a = 0.4, double x = 3.0,
                                 double b = 2.0, double c = 5.0, double m = 3.0) {
  using AD = xad::adj<double>::active_type;

  // Value and (d/da, d/dx) in one reverse sweep.
  double value, d_da, d_dx;
  {
    xad::adj<double>::tape_type tape;
    AD A = a, X = x;
    tape.registerInput(A);
    tape.registerInput(X);
    tape.newRecording();
    AD v = odelia::incomplete_gamma(A, X);
    tape.registerOutput(v);
    xad::derivative(v) = 1.0;
    tape.computeAdjoints();
    value = xad::value(v);
    d_da = xad::derivative(A);
    d_dx = xad::derivative(X);
  }

  // Weibull composition G(m) = (b/c) gamma(1/c, (m/b)^c); (dG/dm, dG/dc).
  double G, dG_dm, dG_dc;
  {
    using std::exp;
    using std::log;
    xad::adj<double>::tape_type tape;
    AD M = m, C = c;
    tape.registerInput(M);
    tape.registerInput(C);
    tape.newRecording();
    AD Xc = exp(C * log(M / b));                       // (m/b)^c
    AD inv_c = 1.0 / C;
    AD Gv = (b / C) * odelia::incomplete_gamma(inv_c, Xc);
    tape.registerOutput(Gv);
    xad::derivative(Gv) = 1.0;
    tape.computeAdjoints();
    G = xad::value(Gv);
    dG_dm = xad::derivative(M);
    dG_dc = xad::derivative(C);
  }

  const double h = 1e-6;
  const double d_da_fd =
      (odelia::incomplete_gamma(a + h, x) - odelia::incomplete_gamma(a - h, x)) / (2 * h);
  const double dG_dc_fd = (weibull_int(m, b, c + h) - weibull_int(m, b, c - h)) / (2 * h);

  // Large x: the integrand has no mass past a few multiples of a, so the series
  // must still reach gamma(a, x) within the term cap. Checked in R against pgamma.
  const double x_large = 40.0;
  const double value_large = odelia::incomplete_gamma(a, x_large);

  return Rcpp::List::create(
      Rcpp::Named("a") = a, Rcpp::Named("x") = x,
      Rcpp::Named("value") = value,
      Rcpp::Named("x_large") = x_large,
      Rcpp::Named("value_large") = value_large,
      Rcpp::Named("d_dx") = d_dx,
      Rcpp::Named("d_dx_ref") = std::pow(x, a - 1.0) * std::exp(-x),
      Rcpp::Named("d_da") = d_da,
      Rcpp::Named("d_da_fd") = d_da_fd,
      Rcpp::Named("dG_dm") = dG_dm,
      Rcpp::Named("dG_dm_ref") = std::exp(-std::pow(m / b, c)),
      Rcpp::Named("dG_dc") = dG_dc,
      Rcpp::Named("dG_dc_fd") = dG_dc_fd);
}
