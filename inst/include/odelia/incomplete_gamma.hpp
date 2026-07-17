#ifndef ODELIA_INCOMPLETE_GAMMA_HPP_
#define ODELIA_INCOMPLETE_GAMMA_HPP_

#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>

namespace odelia {

// The lower incomplete gamma, gamma(a, x) = integral_0^x t^{a-1} e^{-t} dt for
// a > 0, x >= 0. It is the exact antiderivative behind the stretched-exponential
// (Weibull) integral: with a = 1/c and X = (m/b)^c,
//
//     integral_0^m exp(-(s/b)^c) ds = (b/c) * gamma(1/c, X).
//
// Evaluated by the everywhere-convergent series
//
//     gamma(a, x) = x^a e^{-x} * sum_{n>=0} x^n / (a (a+1) ... (a+n)),
//
// written in elementary operations, so AD reads value, d/dx (which is the
// integrand x^{a-1} e^{-x}, the Leibniz endpoint derivative) and d/da (the shape
// channel a differentiated trait needs) off the same code -- no special-function
// derivative, and none is available from the tape library. The term ratio
// x/(a+n) -> 0; the sum stops once a term is negligible against the running
// total (a value-only decision -- the dropped terms move neither value nor
// derivative). The 1000-term cap bounds x to a few hundred, past the range where
// the integrand has any mass.
template <class S>
S incomplete_gamma(const S& a, const S& x) {
  using std::exp;
  using std::log;
  if (util::to_passive(x) <= 0.0) return S(0.0);
  S term = 1.0 / a;   // n = 0
  S sum = term;
  for (int n = 1; n < 1000; ++n) {
    term *= x / (a + static_cast<double>(n));
    sum += term;
    if (util::to_passive(term) <= 1e-17 * util::to_passive(sum)) break;
  }
  return exp(a * log(x) - x) * sum;
}

}  // namespace odelia

#endif
