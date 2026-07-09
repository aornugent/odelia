#ifndef ODELIA_ODE_FIT_HPP_
#define ODELIA_ODE_FIT_HPP_

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_solver.hpp>
#include <functional>
#include <optional>
#include <type_traits>
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

  std::size_t codomain() const { return 1; }     // a scalar loss

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

// The active inputs a gradient is taken w.r.t.: the parameter indices and the
// initial-state indices to seed. The driver seeds each via the System's
// set_param(i, .) / set_ic(j, .). `values` holds the seed values, params first
// then ics, so it is parallel to the Jacobian columns in that order.
struct DifferentiationTargets {
  std::vector<int>    params;   // param indices to seed active
  std::vector<int>    ics;      // initial-state indices to seed active
  std::vector<double> values;   // seed values, params-then-ics

  bool empty() const { return params.empty() && ics.empty(); }
  std::size_t size() const { return params.size() + ics.size(); }
};

// Reverse-mode Jacobian (row i = d(output_i)/d(input)) of a multi-output
// functional w.r.t. the seeded inputs. The record-once/row-sweep is
// xad::computeJacobian's; the forward callback here is the seam it doesn't know:
// seed the registered inputs onto the System (set_param / set_ic), replay the
// recorded `schedule` (advance_fixed), hand the positioned solver to the functional.
// Owning the
// replay here keeps the grid from drifting from the recording, and `schedule.empty()`
// is the whole "forgot to record" guard.
//
// The functional reports its own output count via functional.codomain(); passing it
// to XAD avoids an extra forward callback just to size the output (a wasted full
// model replay), and reading it off the functional means the count and the outputs
// it returns cannot silently disagree.
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_jacobian(
    Solver& solver,
    const DifferentiationTargets& targets,
    const std::vector<double>& schedule,
    Functional&& functional
) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;
    const std::size_t codomain = functional.codomain();

    if (targets.empty()) {
        util::stop("DifferentiationTargets must seed at least one input");
    }
    if (targets.size() != targets.values.size()) {
        util::stop("DifferentiationTargets: 'values' must match 'params' + 'ics'");
    }
    if (schedule.empty()) {
        util::stop("no recorded schedule to replay; run the adaptive pass first");
    }

    // Reuse tape across calls to avoid invalidating registered inputs
    if (!solver.tape) {
        solver.tape = std::make_unique<ad::tape_type>(false); // create without activating
    }
    solver.tape->activate();
    detail::tape_deactivate_guard<ad::tape_type> guard{solver.tape.get()};

    // xad::computeJacobian owns the input registration, so it takes the inputs as
    // one flat vector and calls the forward callback to map them onto the System.
    // The inputs are ordered params-then-ics, matching `values`; the driver seeds
    // each via the System's indexed setters.
    std::vector<ad_type> inputs(targets.values.begin(), targets.values.end());

    std::vector<double> values;
    std::function<std::vector<ad_type>(std::vector<ad_type>&)> forward =
        [&](std::vector<ad_type>& x) {
            auto& sys = solver.get_system_ref();
            std::size_t k = 0;
            for (int i : targets.params) sys.set_param(i, x[k++]);
            for (int j : targets.ics)    sys.set_ic(j, x[k++]);
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

// Adapts a scalar functional into a one-output functional (codomain() == 1) for
// the Jacobian driver.
template<typename Functional>
struct scalar_functional {
  Functional f;
  std::size_t codomain() const { return 1; }
  template<typename Solver>
  std::vector<typename Solver::value_type> operator()(Solver& s) const {
    return { f(s) };
  }
};

// Reverse-mode gradient of a scalar `functional(solver)` w.r.t. the seeded inputs.
// A gradient is the one-row Jacobian of a scalar functional -- record once, one
// sweep -- so this is a thin adapter over the one driver.
template<typename Solver, typename Functional>
std::pair<double, std::vector<double>> compute_gradient(
    Solver& solver,
    const DifferentiationTargets& targets,
    const std::vector<double>& schedule,
    Functional&& functional
) {
    scalar_functional<std::decay_t<Functional>> one{std::forward<Functional>(functional)};
    auto [values, jacobian] = compute_jacobian(solver, targets, schedule, one);
    return {values[0], jacobian[0]};
}

} // namespace ode
} // namespace odelia

#endif
