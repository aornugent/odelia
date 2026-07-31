// -*-c++-*-
#ifndef ODELIA_IMPLICIT_NODE_HPP_
#define ODELIA_IMPLICIT_NODE_HPP_

#include <cmath>
#include <type_traits>
#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

namespace odelia {

// Which way the residual's own derivative points at the operating point:
// dF/dy > 0 (a rising balance) or dF/dy < 0 (a maximiser, whose gradient falls
// through zero). `any` skips the check, for the callers whose sign is not in
// question.
enum class denom_sign { positive, negative, any };

// The value y* defined implicitly by a scalar equation F(y; p) = 0, made
// differentiable. y* is solved in double, off the tape, by whatever root-find the
// caller already has; this returns it on the tape carrying the derivative the
// implicit function theorem gives:
//     dy*/dp = -(dF/dp) / (dF/dy).
// F is a callable F(S y) evaluating the equation at the active parameters, which it
// reaches through its enclosing scope rather than an enumerated vector; dF/dy is a
// double central difference at y*. The returned value is exactly y* bit for bit, so
// a quantity no active parameter reaches introduces no shift, and a plain-double S
// returns y* with nothing recorded. Reverse, forward and nested forward-over-reverse
// scalars all thread dy*/dp through the graft below.
//
// `expect` guards invertibility: at a fold dF/dy -> 0 and this divides by ~0.
// Declaring the operating point's sign turns that into a stop rather than a silent
// garbage gradient.
template <class S, class Equation>
S implicit_value(double y_star, Equation&& F, denom_sign expect = denom_sign::any) {
  // A lambda written `[&](S y) { return ...; }` deduces an XAD expression-template
  // return type holding references to the temporaries of its return statement;
  // those die when it returns, so evaluating F here would read a destroyed tape
  // slot -- a segfault on the reverse sweep, not a wrong number. Requiring S
  // exactly makes that a compile error.
  static_assert(std::is_same_v<std::invoke_result_t<Equation&, S>, S>,
                "implicit_value: the residual callable must return S exactly -- "
                "declare the lambda's return type (e.g. [&](S y) -> S { ... }). "
                "A deduced return type is an expression template referencing "
                "temporaries that are dead by the time this evaluates it.");
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    const double eps = 1e-6 * (std::abs(y_star) + 1.0);
    // The central difference is a pure double probe. If F builds tape nodes of its
    // own -- a nested implicit_value, say -- they must not be recorded here: their
    // values are discarded, so they would be orphan nodes on the active tape, and
    // probe x difference x nesting corrupts it. Stop recording across the probe;
    // the one derivative-carrying evaluation of F is the graft below.
    auto* tape = xad::Tape<double>::getActive();
    const bool was_recording = (tape != nullptr) && tape->isActive();
    if (was_recording) tape->deactivate();
    auto Fd = [&](double y) -> double { return util::to_passive(F(S(y))); };
    const double dFdy = (Fd(y_star + eps) - Fd(y_star - eps)) / (2.0 * eps);
    if (was_recording) tape->activate();
    if (expect != denom_sign::any) {
      const bool ok = (expect == denom_sign::positive) ? (dFdy > 0.0) : (dFdy < 0.0);
      if (!ok)
        util::stop("implicit_value: dF/dy has the wrong sign -- the operating "
                   "point is not invertible (near a fold?)");
    }
    // corr's value is ~0, since y* is the root; its derivative is (dF/dp)/(dF/dy).
    // Subtracting its own passive value leaves the returned value exactly y* and
    // the derivative -corr' = dy*/dp. to_passive strips every layer, so this
    // composes at a nested scalar too.
    const S corr = F(S(y_star)) / dFdy;
    return S(y_star) - corr + util::to_passive(corr);
  }
}

}

#endif
