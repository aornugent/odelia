/* Exercises the scalar value guards util::smooth_positive and util::is_finite,
 * compiled on demand by test-ad-value-guards.R (sourceCpp).
 *
 * smooth_positive is a C-infinity max(0, x) with a declared corner radius; the
 * demo returns its value against max(0, x) in the small-radius limit, its exact
 * derivative, and the forward-vs-reverse dot-product oracle. is_finite reads an
 * active value without a strip, so it reports NaN on the active type.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace Rcpp;

template <class S>
static S reduce(const std::vector<S>& x, const std::vector<double>& w, double r) {
  S total(0.0);
  for (std::size_t i = 0; i < x.size(); ++i) total += w[i] * odelia::util::smooth_positive(x[i], r);
  return total;
}

// [[Rcpp::export]]
Rcpp::List value_guards_demo(double r = 1e-2) {
  std::vector<double> x = {-2.0, -0.5, 0.0, 0.5, 2.0};
  std::vector<double> w = {0.6, 0.8, 1.0, 1.2, 1.4};
  const std::size_t n = x.size();

  // Value vs the sharp max(0, x) in the small-radius limit (r -> 0).
  std::vector<double> sp_tiny, relu;
  for (double xi : x) {
    sp_tiny.push_back(odelia::util::smooth_positive(xi, 1e-8));
    relu.push_back(xi > 0.0 ? xi : 0.0);
  }

  // Reverse gradient of sum_i w_i smooth_positive(x_i, r) vs the analytic
  // derivative 0.5 (1 + x / sqrt(x^2 + r^2)).
  using ad = xad::adj<double>;
  using AD = ad::active_type;
  ad::tape_type tape;
  std::vector<AD> xa(x.begin(), x.end());
  for (auto& v : xa) tape.registerInput(v);
  tape.newRecording();
  AD F = reduce<AD>(xa, w, r);
  tape.registerOutput(F);
  xad::derivative(F) = 1.0;
  tape.computeAdjoints();
  std::vector<double> grad(n), grad_ref(n);
  for (std::size_t i = 0; i < n; ++i) {
    grad[i] = xad::derivative(xa[i]);
    grad_ref[i] = w[i] * 0.5 * (1.0 + x[i] / std::sqrt(x[i] * x[i] + r * r));
  }

  // Dot-product oracle.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::vector<double> dir(n);
  for (auto& d : dir) d = unif(rng);
  using FAD = xad::fwd<double>::active_type;
  std::vector<FAD> xf(x.begin(), x.end());
  for (std::size_t i = 0; i < n; ++i) xad::derivative(xf[i]) = dir[i];
  FAD Ff = reduce<FAD>(xf, w, r);
  double dot = 0.0;
  for (std::size_t i = 0; i < n; ++i) dot += grad[i] * dir[i];

  // is_finite on the active type: a NaN reports false, a finite value true,
  // with no strip.
  AD nan_active = AD(0.0) / AD(0.0);
  AD finite_active = AD(3.0);
  bool nan_is_finite = odelia::util::is_finite(nan_active);
  bool finite_is_finite = odelia::util::is_finite(finite_active);

  return Rcpp::List::create(
      Rcpp::Named("sp_tiny") = wrap(sp_tiny),
      Rcpp::Named("relu") = wrap(relu),
      Rcpp::Named("grad") = wrap(grad),
      Rcpp::Named("grad_ref") = wrap(grad_ref),
      Rcpp::Named("jvp") = xad::derivative(Ff),
      Rcpp::Named("dot_v_grad") = dot,
      Rcpp::Named("nan_is_finite") = nan_is_finite,
      Rcpp::Named("finite_is_finite") = finite_is_finite);
}
