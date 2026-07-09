#ifndef ODELIA_ODE_FIT_HPP_
#define ODELIA_ODE_FIT_HPP_

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_solver.hpp>
#include <functional>
#include <optional>
#include <vector>

namespace odelia {
namespace ode {

// Sum-of-squares between predicted and measured observations; the reduction of
// least_squares, split out so it can be reused and tested directly.
template<typename T>
T sum_of_squares(const std::vector<std::vector<T>>& predicted,
                 const std::vector<std::vector<double>>& observations) {
    T loss(0.0);
    for (size_t i = 0; i < predicted.size(); ++i) {
        for (size_t j = 0; j < predicted[i].size(); ++j) {
            T diff = predicted[i][j] - observations[i][j];
            loss += diff * diff;
        }
    }
    return loss;
}

// A functional is a pure reduction: it reads an already-replayed Solver and
// returns the scalar(s) to differentiate. It does not drive the solver or carry a
// schedule -- the driver replays, the functional reduces. The scalar is anything:
// a calibration loss, an emergent summary of the state (a stand's basal area).
//
// A calibration functional: holds measured data, scores the replayed trajectory
// against it. The solver stores no fit state. The sampling grid is the recording's
// (recorded_steps()), so obs_indices index into it -- least_squares carries no
// schedule of its own. It reads interior points from the solver's collected
// history (bare history rows are #23).
struct least_squares {
  std::vector<size_t>              obs_indices;   // indices into recorded_steps()
  std::vector<std::vector<double>> observations;  // measured data, per obs point

  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    using value_type = typename Solver::value_type;
    std::vector<std::vector<value_type>> predicted;
    predicted.reserve(obs_indices.size());
    for (size_t idx : obs_indices) {
      auto sys = solver.get_history_step(idx);
      std::vector<value_type> s(sys.ode_size());
      sys.ode_state(s.begin());
      predicted.push_back(std::move(s));
    }
    return sum_of_squares(predicted, observations);
  }
};

namespace detail {
// Deactivate the tape on every exit path -- normal return or any exception.
template<typename Tape>
struct tape_deactivate_guard {
    Tape* tape;
    ~tape_deactivate_guard() { tape->deactivate(); }
};
} // namespace detail

// The active inputs a gradient is taken w.r.t.: opaque (slot, value) leaves. The
// System owns the slot->field routing (its scatter method), so odelia never learns
// whether a leaf is a trait, an initial density, or a flux. Jacobian column j
// corresponds to (slots[j], values[j]).
struct DifferentiationTargets {
  std::vector<int>    slots;
  std::vector<double> values;   // parallel to slots

  bool empty() const { return values.empty(); }
};

// Reverse-mode Jacobian (row i = d(output_i)/d(input)) of a multi-output
// functional w.r.t. the seeded inputs. The record-once/row-sweep is
// xad::computeJacobian's; the forward callback here is the seam it doesn't know:
// scatter the registered inputs onto the System, replay the recorded `schedule`
// (advance_fixed), and hand the positioned solver to the functional. Owning the
// replay here keeps the grid from drifting from the recording, and `schedule.empty()`
// is the whole "forgot to record" guard.
//
// `codomain` is the number of outputs m: pass it, or XAD runs the forward callback
// an extra time just to size the output (a wasted full model replay).
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_jacobian(
    Solver& solver,
    const DifferentiationTargets& independents,
    const std::vector<double>& schedule,
    Functional&& functional,
    std::size_t codomain
) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    if (independents.empty()) {
        util::stop("DifferentiationTargets must seed at least one leaf");
    }
    if (independents.slots.size() != independents.values.size()) {
        util::stop("DifferentiationTargets: 'slots' and 'values' must be the same length");
    }
    if (schedule.empty()) {
        util::stop("no recorded schedule to replay; run the adaptive pass first");
    }

    // Reuse tape across calls to avoid invalidating slots
    if (!solver.tape) {
        solver.tape = std::make_unique<ad::tape_type>(false); // create without activating
    }
    solver.tape->activate();
    detail::tape_deactivate_guard<ad::tape_type> guard{solver.tape.get()};

    // xad::computeJacobian owns the input registration, so it takes the inputs as
    // one flat vector and calls the forward callback to map them onto the System;
    // the System's scatter routes each active value to the field its slot names.
    std::vector<ad_type> inputs(independents.values.begin(), independents.values.end());

    std::vector<double> values;
    std::function<std::vector<ad_type>(std::vector<ad_type>&)> forward =
        [&](std::vector<ad_type>& x) {
            solver.get_system_ref().scatter(x.begin(), independents.slots);
            solver.reset();
            solver.advance_fixed(schedule);        // the driver owns the L1 replay
            auto outputs = functional(solver);     // the functional is a pure reduction
            values.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                values[i] = xad::value(outputs[i]);
            }
            return outputs;
        };

    auto jacobian = xad::computeJacobian(inputs, forward, codomain, solver.tape.get());
    return {values, jacobian};
}

// Reverse-mode gradient of a scalar `functional(solver)` w.r.t. the active
// inputs named by `independents`. A gradient is the one-row Jacobian of a scalar
// functional -- in adjoint mode with codomain = 1 the cost is identical (record
// once, one sweep) -- so this is a thin adapter over the one driver: wrap the
// scalar functional into a 1-vector functional and unwrap the single row.
template<typename Solver, typename Functional>
std::pair<double, std::vector<double>> compute_gradient(
    Solver& solver,
    const DifferentiationTargets& independents,
    const std::vector<double>& schedule,
    Functional&& functional
) {
    using value_type = typename Solver::value_type;
    auto as_vector = [&](Solver& s) {
        return std::vector<value_type>{ functional(s) };
    };
    auto [values, jacobian] = compute_jacobian(solver, independents, schedule, as_vector, 1);
    return {values[0], jacobian[0]};
}

} // namespace ode
} // namespace odelia

#endif
