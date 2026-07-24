#ifndef ODELIA_IMPLICIT_NODE_HPP_
#define ODELIA_IMPLICIT_NODE_HPP_

#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>
#include <type_traits>

namespace odelia {

// Which way the residual's own derivative points at the operating point: dF/dy > 0
// (a rising balance) or dF/dy < 0 (a maximiser, where the objective's gradient falls
// through zero). Passed to implicit_value to assert invertibility; `any` skips the
// check (the default, for the many callers whose sign is not in question).
enum class denom_sign { positive, negative, any };

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
// spurious shift; a plain-double S returns y* with nothing recorded. It composes
// across scalar modes: reverse (adjoint), forward (tangent), and nested
// forward-over-reverse all thread dy*/dp correctly through the graft idiom below.
//
// `expect` guards invertibility: at a genuine fold dF/dy -> 0 and this node would
// divide by ~0 (the b1 blow-up). Declaring the operating point's sign makes that a
// loud stop instead of a silent garbage gradient; leave it `any` when the sign is
// not in doubt.
template <class S, class Equation>
S implicit_value(double y_star, Equation&& F, denom_sign expect = denom_sign::any) {
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    const double eps = 1e-6 * (std::abs(y_star) + 1.0);
    // The dF/dy central difference is a pure double probe. If F internally builds
    // tape nodes (e.g. a nested implicit_value, as in a re-optimised leaf), those
    // must NOT be recorded here: their value is discarded by to_passive, so they
    // would be orphan nodes on the active tape, and accumulating them (probe x
    // central-difference x nesting) corrupts the tape (a segfault in XAD's operand
    // push at depth). Pause recording across the probe; the single derivative-
    // carrying evaluation of F is the graft below.
    auto* tape = xad::Tape<double>::getActive();
    const bool was_recording = (tape != nullptr) && tape->isActive();
    if (was_recording) tape->deactivate();
    auto Fd = [&](double y) { return util::to_passive(F(S(y))); };
    const double dFdy = (Fd(y_star + eps) - Fd(y_star - eps)) / (2.0 * eps);
    if (was_recording) tape->activate();
    if (expect != denom_sign::any) {
      const bool ok = (expect == denom_sign::positive) ? (dFdy > 0.0) : (dFdy < 0.0);
      if (!ok)
        util::stop("implicit_value: dF/dy has the wrong sign -- the operating point is not invertible (near a fold?)");
    }
    // corr's value is ~0 (y* is the root); its derivative is (dF/dp)/(dF/dy).
    // Subtracting its own passive value keeps the returned value exactly y* while
    // leaving the derivative as -corr' = dy*/dp. to_passive (not xad::value) strips
    // every layer, so this composes at a nested forward-over-reverse S too.
    const S corr = F(S(y_star)) / dFdy;
    return S(y_star) - corr + util::to_passive(corr);
  }
}

}  // namespace odelia

#endif
