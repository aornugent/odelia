#ifndef ODELIA_PREACCUMULATE_HPP_
#define ODELIA_PREACCUMULATE_HPP_

#include <XAD/XAD.hpp>
#include <odelia/ode_util.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace odelia {

// Record a block's local derivatives instead of its internals.
//
// A block with many operations and ONE output -- a fixed-rule quadrature, a
// multi-term series, a crown integral -- costs its whole operation count on the run
// tape, while all the reverse sweep can ever extract from it is one partial per
// input. Evaluate it on a tape of its own instead, sweep that once, and put only the
// partials on the run tape: the cost becomes n_inputs, independent of how much work
// the block does.
//
// f is a pure function of the declared inputs -- it receives them as a vector and
// must not read active values from enclosing scope. That is what makes the input list
// checkable rather than a convention: a channel omitted from `inputs` cannot be
// reached from inside f at all, so it fails as a compile error or a zero, not as a
// plausible gradient missing one term.
//
// The partials are computed by AD, not by hand. This is a tape-size transform, so it
// adds no hand-written reverse rule and leaves the count of those unchanged.
//
// FIRST ORDER. The injected form carries exact first derivatives and zero curvature,
// so a second-derivative consumer must call the block directly instead. Everything
// preaccumulated is second-order blind by construction; keep such call sites
// enumerable.
template <class S, class F>
S preaccumulate(const std::vector<S>& inputs, F&& f) {
  if constexpr (std::is_same_v<S, double>) {
    return f(inputs);
  } else {
    using tape_type = typename xad::adj<double>::tape_type;
    static_assert(std::is_same_v<decltype(f(inputs)), S>,
                  "preaccumulate: the block must return S exactly -- declare the "
                  "callable's return type (e.g. [](const std::vector<S>& x) -> S). A "
                  "deduced return type is an expression template referencing "
                  "temporaries that are dead by the time this evaluates it.");

    const std::size_t n = inputs.size();
    std::vector<double> partials(n, 0.0);
    double value = 0.0;

    // The block runs on its own tape, so nothing it does reaches the run tape. The
    // run tape must be inactive across that: leaving it recording would put the
    // block's internals on it, which is the cost being avoided.
    auto* outer = xad::Tape<double>::getActive();
    const bool was_recording = (outer != nullptr) && outer->isActive();
    if (was_recording) outer->deactivate();
    {
      tape_type inner;
      std::vector<S> x(n);
      for (std::size_t i = 0; i < n; ++i) x[i] = S(util::to_passive(inputs[i]));
      for (std::size_t i = 0; i < n; ++i) inner.registerInput(x[i]);
      inner.newRecording();
      S out = f(x);
      inner.registerOutput(out);
      xad::derivative(out) = 1.0;
      inner.computeAdjoints();
      value = xad::value(out);
      for (std::size_t i = 0; i < n; ++i) partials[i] = xad::derivative(x[i]);
    }
    if (was_recording) outer->activate();

    // What reaches the run tape: the block's value, and one multiply-add per input.
    S result(value);
    for (std::size_t i = 0; i < n; ++i)
      result = result + partials[i] * (inputs[i] - util::to_passive(inputs[i]));
    return result;
  }
}

}  // namespace odelia

#endif
