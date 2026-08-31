// -*-c++-*-
#ifndef ODELIA_IMPLICIT_NODE_HPP_
#define ODELIA_IMPLICIT_NODE_HPP_

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <XAD/XAD.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_util.hpp>

namespace odelia {

// One input a value responds to, and how. Kept as a pair so the two cannot be
// assembled from separate lists and paired by position.
//
// THE INPUT IS BORROWED, and must outlive the record it is handed to. Held by
// reference because copying an active scalar is not free: the copy registers a
// tape slot and records a statement, for a value that is only ever read: n
// inputs would cost n of those per row.
template <class S>
struct input_and_derivative {
  const S& input;
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
// The value y* defined implicitly by a scalar equation F(y) = 0, made
// differentiable. y* is solved in double, off the tape, by whatever root-find the
// caller already has; this returns it on the tape carrying the derivative the
// implicit function theorem gives:
//     dy*/dp = -(dF/dp) / (dF/dy).
//
// dF/dp is taped: F is evaluated once, at S, and its derivative in every active
// input the residual reads IS dF/dp. dF/dy is the caller's, because a residual
// mostly knows its own slope: it is
// one term of the expression the caller just wrote. One that does not can take a
// tangent through its own body and hand the answer here, which is a local three
// lines rather than a rebindable parameter object every caller must carry.
//
// dF/dy is read at the point, so its own dependence on p belongs to the second
// derivative rather than to this one. It is what the theorem divides by, so a fold
// -- where it approaches zero and the quotient is garbage rather than large -- is
// the one thing this stops on.
//
// ⚠️ THIS ONCE HAD A REPORTING SIBLING, and the sibling went because the distinction
// was one nothing acted on. The argument for it was that a BOUND must stop -- its
// value IS what the equation defines -- while an INTERIOR optimum could carry on with
// the row missing, since the envelope theorem spares the objective. Both were true of
// the theorem and neither was true of the consumer: the report's one reader turned it
// straight into the same metric-level refusal the catch around this makes. Two
// mechanisms, one outcome, and a record_report threaded through LeafOutputs and two
// signatures to carry the difference.
template <class S, class Residual>
S implicit_value(double y_star, double dFdy, Residual&& F) {
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    // A residual written `[](auto y) { return ...; }` deduces an XAD
    // expression-template return type holding references to the temporaries of
    // its return statement; those die when it returns, so evaluating it here
    // would read a destroyed tape slot -- a segfault on the reverse sweep, not a
    // wrong number. Requiring the scalar back exactly makes that a compile error.
    static_assert(
        std::is_same_v<std::invoke_result_t<Residual&, const S&>, S>,
        "implicit_value: the residual must return its own scalar exactly "
        "([](const S& y) -> S { ... }). A deduced return type is an expression "
        "template referencing temporaries that are dead by the time this "
        "evaluates it.");
    if (!util::is_finite(dFdy) || dFdy == 0.0) {
      util::stop("implicit_value: dF/dy is " + util::format_double(dFdy) +
                 " at the operating point, so the implicit function theorem "
                 "does not apply there (a fold?)");
    }
    // corr's value is ~0, since y* is the root; its derivative is (dF/dp)/(dF/dy),
    // so y* against it with a coefficient of -1 is the theorem's own quotient.
    std::vector<input_and_derivative<S>> corrections;
    const S corr = F(S(y_star)) / dFdy;
    corrections.push_back({corr, -1.0});
    S out;
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
