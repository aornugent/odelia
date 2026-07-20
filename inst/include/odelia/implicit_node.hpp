#ifndef ODELIA_IMPLICIT_NODE_HPP_
#define ODELIA_IMPLICIT_NODE_HPP_

#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>
#include <odelia/supplied_derivative.hpp>

#include <cmath>
#include <type_traits>
#include <vector>

namespace odelia {

template <class> inline constexpr bool always_false = false;

// The value y* defined implicitly by a scalar equation F(y; p) = 0, made
// differentiable. y* is solved OFF the tape (in double, e.g. by a root-find); this
// returns it on the tape carrying the derivative the implicit function theorem gives
// for the root of an equation:
//     dy*/dp = -(dF/dp) / (dF/dy).
// F is a callable F(S y) that evaluates the equation at the active parameters,
// reaching them through its enclosing scope (not an enumerated vector); dF/dy is a
// double central difference at y*. This suits a System that reads its own parameter
// members: the argument is just the defining equation written in the working scalar,
// and no AD machinery appears in the caller. The returned value is exactly y*
// (bit-identical), so a quantity other parameters do not depend on introduces no
// spurious shift; a plain-double S returns y* with nothing recorded.
//
// Sibling of register_implicit -- both give the implicit-function-theorem derivative
// of a solved value. Use register_implicit when the reacting inputs are a known
// vector and F cannot read them from scope (it forward-differentiates F per input and
// injects via supplied_derivative); use implicit_value when F reads the active inputs
// directly, the common case for a Strategy.
template <class S, class Equation>
S implicit_value(double y_star, Equation&& F) {
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    const double eps = 1e-6 * (std::abs(y_star) + 1.0);
    auto Fd = [&](double y) { return util::to_passive(F(S(y))); };
    const double dFdy = (Fd(y_star + eps) - Fd(y_star - eps)) / (2.0 * eps);
    // corr's value is ~0 (y* is the root); its derivative is (dF/dp)/(dF/dy).
    // Subtracting its own passive value keeps the returned value exactly y* while
    // leaving the derivative as -corr' = dy*/dp. to_passive (not xad::value) strips
    // every layer, so this composes at a nested forward-over-reverse S too.
    const S corr = F(S(y_star)) / dFdy;
    return S(y_star) - corr + util::to_passive(corr);
  }
}

// Which way the residual's own derivative must point at the operating point:
// dF/dy > 0 (a rising balance) or dF/dy < 0 (a maximiser, where the objective's
// gradient falls through zero). Asserted at registration.
enum class denom_sign { positive, negative };

// A scalar inner solve F(y; p) = 0 whose root is found OFF the tape, with its
// derivative supplied by the implicit function theorem instead of by recording
// the iteration:
//
//     dy/dp_i = -(dF/dp_i) / (dF/dy)   at the operating point.
//
// F is scalar-generic -- F(y, p) with y a scalar and p a vector of scalars of
// the working type -- and depends on the tape only through p; every active
// input the root reacts to is passed in p, all else being plain double. `solve`
// finds y* from the double parameter values. register_implicit forms the
// partials by forward-differentiating F once per input at (y*, p), asserts the
// sign of dF/dy, and carries dy/dp through whichever mode S is: a reverse
// adjoint (via supplied_derivative), a forward tangent, or nothing for a plain
// double. The iteration is never recorded, so no nested tape is needed.
template <class S, class Residual, class Solve>
S register_implicit(Residual&& F, Solve&& solve, const std::vector<S*>& p,
                    denom_sign expect) {
  using fwd = typename xad::fwd<double>::active_type;
  const std::size_t np = p.size();

  std::vector<double> pv(np);
  for (std::size_t i = 0; i < np; ++i) pv[i] = util::to_passive(*p[i]);
  const double y0 = solve(pv);

  // One forward pass of F gives its derivative along a chosen direction in
  // (y, p); seed y alone for dF/dy, then each p_k alone for dF/dp_k.
  auto directional = [&](double dy, const std::vector<double>& dp) {
    fwd y(y0);
    xad::derivative(y) = dy;
    std::vector<fwd> pf(np);
    for (std::size_t i = 0; i < np; ++i) {
      pf[i] = fwd(pv[i]);
      xad::derivative(pf[i]) = dp[i];
    }
    return xad::derivative(F(y, pf));
  };

  const std::vector<double> none(np, 0.0);
  const double dFdy = directional(1.0, none);
  const bool ok = (expect == denom_sign::positive) ? (dFdy > 0.0) : (dFdy < 0.0);
  if (!ok) util::stop("register_implicit: dF/dy has the wrong sign -- the operating point is not invertible");

  std::vector<double> dydp(np);
  for (std::size_t k = 0; k < np; ++k) {
    std::vector<double> e(np, 0.0);
    e[k] = 1.0;
    dydp[k] = -directional(0.0, e) / dFdy;
  }

  using rev = typename xad::adj<double>::active_type;
  if constexpr (std::is_same_v<S, double>) {
    return y0;
  } else if constexpr (std::is_same_v<S, fwd>) {
    S y(y0);
    double tangent = 0.0;
    for (std::size_t i = 0; i < np; ++i) tangent += dydp[i] * xad::derivative(*p[i]);
    xad::derivative(y) = tangent;
    return y;
  } else if constexpr (std::is_same_v<S, rev>) {
    auto* tape = xad::adj<double>::tape_type::getActive();
    if (tape == nullptr) util::stop("register_implicit: no active tape for the reverse injection");
    return ode::supplied_derivative(*tape, y0, p, dydp);
  } else {
    // First-order only: a nested type (tangent-over-adjoint) would need a
    // second-order rule, which this node does not supply -- fail loudly rather
    // than take the reverse path silently.
    static_assert(always_false<S>,
                  "register_implicit supports scalar double, forward, and reverse modes only");
  }
}

}  // namespace odelia

#endif
