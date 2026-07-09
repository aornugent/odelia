// -*-c++-*-
#ifndef ODELIA_SUPPLIED_DERIVATIVE_HPP_
#define ODELIA_SUPPLIED_DERIVATIVE_HPP_

#include <XAD/XAD.hpp>
#include <XAD/CheckpointCallback.hpp>
#include <odelia/ode_util.hpp>

#include <vector>

namespace odelia {
namespace ode {

// One reverse-mode edge with analytic partials, injected through XAD's
// CheckpointCallback. When the reverse sweep reaches the edge it reads the
// adjoint accumulated on the output slot and distributes it to the input slots
// weighted by dy/dx_i -- no forward operations of the edge were recorded.
template <typename Tape>
class SuppliedDerivative : public xad::CheckpointCallback<Tape> {
  using slot_type = typename Tape::slot_type;
public:
  SuppliedDerivative(slot_type output, std::vector<slot_type> inputs,
               std::vector<double> partials)
    : output_(output), inputs_(std::move(inputs)), partials_(std::move(partials)) {}

  void computeAdjoint(Tape* tape) override {
    const auto ybar = tape->getAndResetOutputAdjoint(output_);
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      tape->incrementAdjoint(inputs_[i], ybar * partials_[i]);
    }
  }

private:
  slot_type output_;
  std::vector<slot_type> inputs_;
  std::vector<double> partials_;
};

// Make an off-tape result active, carrying its known partials into the reverse
// tape. `y_value` is a quantity the forward pass computed *off* the tape -- a
// root-find, an optimiser result -- and `partials[i]` is the analytic dy/dx_i
// w.r.t. the active `inputs` (the implicit-function-theorem edge). The internal
// solve is not recorded; instead `y` becomes a fresh tape leaf and an
// SuppliedDerivative distributes its adjoint to the inputs on the reverse sweep.
//
// This is odelia's generic seam for plant's forward-mode leaf optimiser and the
// TF24 stomatal IFT -- one mechanism replacing per-model hand-rolled injections
// (cf. the spike's inject_h0). odelia sees only inputs, an output value, and
// partials. The tape takes ownership of the edge (freed with the tape).
template <typename Tape, typename Active>
Active supplied_derivative(Tape& tape, double y_value,
                     const std::vector<Active*>& inputs,
                     const std::vector<double>& partials) {
  util::check_length(partials.size(), inputs.size());

  Active y = y_value;
  tape.registerInput(y);  // a fresh leaf; downstream operations record its slot

  std::vector<typename Tape::slot_type> input_slots;
  input_slots.reserve(inputs.size());
  for (const auto* x : inputs) {
    input_slots.push_back(x->getSlot());
  }

  auto* edge = new SuppliedDerivative<Tape>(y.getSlot(), std::move(input_slots), partials);
  tape.pushCallback(edge);    // hand ownership to the tape
  tape.insertCallback(edge);  // place the edge at the current tape position
  return y;
}

} // namespace ode
} // namespace odelia

#endif
