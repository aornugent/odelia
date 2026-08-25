// -*-c++-*-
#ifndef ODELIA_IMPLICIT_NODE_HPP_
#define ODELIA_IMPLICIT_NODE_HPP_

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#include <XAD/XAD.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_util.hpp>
#include <odelia/tangent.hpp>

namespace odelia {

// One input a value responds to, and how. Kept as a pair so the two cannot be
// assembled from separate lists and paired by position.
template <class S>
struct input_and_derivative {
  S input;
  double derivative;
};

// What a record could not record, beside the value it wrote. A row that cannot be
// recorded is not an error here: the value is still the value, and whether a
// consumer can go on without the row is the consumer's to decide -- one output's
// rows can go missing while another's survive, and a stop takes both.
struct record_report {
  bool whole = true;
  // Which input's row is missing, and what was wrong with it. Meaningless where
  // `whole`.
  std::size_t at = 0;
  std::string why;
};

// `into` receives `value` carrying the derivatives supplied against it: the
// number is `value` itself, and its derivative with respect to each input is the
// one supplied. Each term is an input minus its own passive copy, which is zero
// in value and carries the derivative, so the number is untouched and only the
// tape sees the terms.
//
// This is how a quantity computed away from the tape gets onto it -- a
// root-find, a submodel's own solve, anything whose derivative is known by some
// means other than recording the steps that produced it. At a plain double the
// terms all vanish and this is the value.
//
// NOTHING PARTIAL. Every row is tested before any is recorded, because a value
// carrying some of its rows is a channel that has gone missing with every number
// still finite -- which is worse than no rows at all, since a consumer told the
// rows are absent can carry the value as a constant and say so.
//
// The report is the return, so a caller cannot take the number without being
// handed the reading of it.
template <class S>
[[nodiscard]] record_report record_with_derivatives(
    double value, const std::vector<input_and_derivative<S>>& against, S& into) {
  for (std::size_t i = 0; i < against.size(); ++i) {
    // Either half being non-finite poisons the VALUE and not only what is
    // recorded against it: NaN times zero is not a number, and an infinite input
    // minus its own passive copy is NaN rather than zero. Tested here, where the
    // halves are still separable; downstream they are one expression.
    const double at = util::to_passive(against[i].input);
    if (!util::is_finite(against[i].derivative) || !util::is_finite(at)) {
      into = value;
      return {false, i,
              "input " + util::to_string(static_cast<int>(i)) + " of " +
                  util::to_string(static_cast<int>(against.size())) +
                  " has value " + util::format_double(at) + " and derivative " +
                  util::format_double(against[i].derivative) +
                  ", one of which is not finite, so the value they belong to "
                  "cannot be recorded"};
    }
  }
  S out = value;
  for (const input_and_derivative<S>& term : against) {
    // A zero derivative contributes exactly zero to the value and exactly nothing
    // to the transpose, and recording it costs a tape edge that the sweep then walks
    // twice.
    if (term.derivative == 0.0) {
      continue;
    }
    out += term.derivative * (term.input - util::to_passive(term.input));
  }
  into = out;
  return {};
}

// The quantity a solve left at a root of R(p; u) = 0, on the tape carrying the
// derivative the implicit function theorem gives:
//     dp/du = -(dR/du) / (dR/dp).
// Both slopes are supplied, because the solve is the caller's and none of it was
// recorded. What this owns is the quotient, its sign, and the refusal where
// dR/dp is not invertible.
//
// Outputs that depend on p record against the value returned here, each carrying
// only its own dy/dp, so the quotient is formed once however many outputs and
// inputs there are. An output the root's own condition makes stationary -- an
// objective at an interior optimum -- records against its inputs and not against
// this, which is the envelope theorem written as an omission rather than as a
// term that has to come out to zero.
template <class S>
[[nodiscard]] record_report implicit_root(
    double p, double residual_slope,
    std::vector<input_and_derivative<S>> against, S& into) {
  // Zero is the only threshold available: how small a slope is too small depends
  // on R's units, which the caller has and this does not. At a fold it
  // approaches zero and the quotient is garbage rather than large. Reported
  // rather than stopped, for the reason the record's own rows are: the point is
  // still the point, and a consumer whose other outputs do not read it can carry
  // on -- where a stop takes those too.
  if (!util::is_finite(residual_slope) || residual_slope == 0.0) {
    into = p;
    return {false, 0,
            "dR/dp is " + util::format_double(residual_slope) +
                " at the operating point, so the implicit function theorem does "
                "not apply there (a fold?)"};
  }
  for (input_and_derivative<S>& term : against) {
    term.derivative /= -residual_slope;
  }
  return record_with_derivatives<S>(p, against, into);
}

// A scalar whose derivative is itself active, which is the only reason to build
// one: a second derivative. It is what decides how many corrections an implicit
// node has to record -- see there.
template <class S>
concept SecondOrder = !std::is_same_v<typename S::derivative_type, double>;

// The value y* defined implicitly by a scalar equation F(y; p) = 0, made
// differentiable. y* is solved in double, off the tape, by whatever root-find the
// caller already has; this returns it on the tape carrying the derivative the
// implicit function theorem gives:
//     dy*/dp = -(dF/dp) / (dF/dy).
//
// F is a template on its scalar, evaluated twice: at a tangent with the
// parameters rebound -- which drops their derivatives, so this is dF/dy exactly --
// and at S with them active, which is dF/dp. `p` holds the residual's ACTIVE
// inputs and nothing else; anything already double is the residual's own to
// capture, and a struct rather than a list because the residual reads them by
// name. The returned value is exactly y* bit for bit, so a quantity no active
// parameter reaches introduces no shift, and a plain-double S returns y* with
// nothing recorded. Reverse, forward and nested forward-over-reverse scalars all
// thread dy*/dp through the record below.
//
// ⚠️ THE DENOMINATOR IS TAKEN ON A SCALAR THAT CANNOT RECORD, and that is why the
// parameters are passed rather than captured. Taken at S and stripped it would put
// every step of the probe on the active tape, and the only defence is a
// deactivate/reactivate bracket that is silent when it is forgotten. Taken as a
// difference it costs two evaluations and 1e-6 of accuracy. A tangent carries
// dF/dy exactly and holds no tape -- and XAD offers no lift from an active scalar
// into a tangent above it, so one evaluation cannot serve both.
//
// dF/dy is read at the point, so its own dependence on p belongs to the second
// derivative rather than to this one.
//
// dF/dy is what the theorem divides by, so a non-invertible operating point stops
// here: at a fold it approaches zero and the quotient is garbage rather than large.
template <class S, class Params, class Residual>
  requires ode::Rebindable<Params, ode::tangent_scalar<double>>
S implicit_value(double y_star, const Params& p, Residual&& F) {
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    using D = ode::tangent_scalar<double>;
    // A residual written `[](auto y, auto p) { return ...; }` deduces an XAD
    // expression-template return type holding references to the temporaries of its
    // return statement; those die when it returns, so evaluating it here would read
    // a destroyed tape slot -- a segfault on the reverse sweep, not a wrong number.
    // Requiring the scalar back exactly makes that a compile error.
    static_assert(
        std::is_same_v<std::invoke_result_t<Residual&, const D&,
                                            decltype(p.template rebind_from<D>())>,
                       D> &&
            std::is_same_v<std::invoke_result_t<Residual&, const S&, const Params&>,
                           S>,
        "implicit_value: the residual must return its own scalar exactly at both "
        "evaluations -- declare the return type "
        "([]<class T>(const T& y, const inputs<T>& p) -> T { ... }). A deduced "
        "return type is an expression template referencing temporaries that are "
        "dead by the time this evaluates it.");
    D probe = y_star;
    ode::seed_direction(probe, 1.0);
    const double dFdy = ode::derivative_along(F(probe, p.template rebind_from<D>()));
    // Zero is the only threshold available here: how small a nonzero slope is too
    // small depends on F's units, which the caller has and this does not.
    if (!util::is_finite(dFdy) || dFdy == 0.0) {
      util::stop("implicit_value: dF/dy is zero at the operating point, so the "
                 "implicit function theorem does not apply there (a fold?)");
    }
    // corr's value is ~0, since y* is the root; its derivative is (dF/dp)/(dF/dy),
    // so y* against it with a coefficient of -1 is the theorem's own quotient.
    // to_passive strips every layer, so this composes at a nested scalar too.
    std::vector<input_and_derivative<S>> corrections;
    const S corr = F(S(y_star), p) / dFdy;
    corrections.push_back({corr, -1.0});
    // ⚠️ ONE CORRECTION IS RIGHT TO FIRST ORDER AND WRONG TO SECOND, because the
    // denominator is a constant: what it records is the theorem linearised, whose
    // curvature is -F_pp/F_y where the true one carries F_yy and F_py too. A
    // second correction taken at the first one's own value supplies exactly those,
    // and its first derivative is zero -- so this changes no gradient, only the
    // curvature of one.
    //
    // Recorded only where the scalar can hold a second derivative: on a plain
    // adjoint it would double the residual's tape to carry something nothing can
    // read.
    if constexpr (SecondOrder<S>) {
      S once;
      const record_report first =
          record_with_derivatives<S>(y_star, corrections, once);
      if (!first.whole) {
        util::stop("implicit_value: " + first.why);
      }
      corrections.push_back({F(once, p) / dFdy, -1.0});
    }
    S out;
    // Stopped rather than reported, and the asymmetry with the two above is the
    // point: this IS the value the equation defines, so a caller handed y* with
    // no derivative has a structural zero and no way to know it. The two above
    // return a value a consumer already has another use for.
    const record_report report =
        record_with_derivatives<S>(y_star, corrections, out);
    if (!report.whole) {
      util::stop("implicit_value: " + report.why);
    }
    return out;
  }
}

}

#endif
