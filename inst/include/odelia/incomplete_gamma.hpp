#ifndef ODELIA_INCOMPLETE_GAMMA_HPP_
#define ODELIA_INCOMPLETE_GAMMA_HPP_

#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>
#include <type_traits>

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
// written in elementary operations. The term ratio x/(a+n) -> 0; the sum stops once
// a term is negligible against the running total (a value-only decision -- the
// dropped terms move neither value nor derivative). The 1000-term cap bounds x to a
// few hundred, past the range where the integrand has any mass.
template <class S>
S gamma_series(const S& a, const S& x) {
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

// The series above converges in tens of terms, so recording it costs tens of tape
// operations per call -- and the hydraulic path calls it per soil layer per leaf
// evaluation. Both partials are available without recording any of it: d/dx is the
// integrand itself, and d/da comes from the same series run in tapeless FORWARD
// mode. So the value and both partials are computed off the tape and injected, and
// what reaches the tape is two multiply-adds instead of the whole sum.
//
// The partials are still computed by AD, not by hand: a shape derivative of this
// function has no elementary closed form, and writing one out would be the kind of
// hand-written reverse rule the rest of the design exists to avoid. Forward mode
// carries no tape, so running the series in it is free of the cost being avoided.
//
// The value is bit-identical to the series, on every mode. The injection is exact
// to first order, which is the order the tape is asked for; second derivatives of
// this function would read as zero, so a second-order consumer must call
// gamma_series directly.
template <class S>
S incomplete_gamma(const S& a, const S& x) {
  if constexpr (std::is_same_v<S, double>) {
    return gamma_series<double>(a, x);
  } else {
    const double a_d = util::to_passive(a), x_d = util::to_passive(x);
    if (x_d <= 0.0) return S(0.0);

    // d/da off the tape: the same series, one forward dual seeded on the shape.
    using dual = typename xad::fwd<double>::active_type;
    dual a_f(a_d);
    xad::derivative(a_f) = 1.0;
    const dual g = gamma_series<dual>(a_f, dual(x_d));

    const double value = xad::value(g);
    const double dg_da = xad::derivative(g);
    const double dg_dx = std::exp((a_d - 1.0) * std::log(x_d) - x_d);  // the integrand

    return S(value) + dg_da * (a - util::to_passive(a)) +
           dg_dx * (x - util::to_passive(x));
  }
}

}  // namespace odelia

#endif
