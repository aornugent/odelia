#ifndef ODELIA_GRADIENT_HPP_
#define ODELIA_GRADIENT_HPP_

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_solver.hpp>
#include <functional>
#include <type_traits>
#include <vector>

namespace odelia {
namespace ode {

// The inputs a gradient is taken with respect to: which parameters and which
// initial-state entries to seed active, and their values. `values` is ordered
// params-then-ics, matching the Jacobian columns.
struct DifferentiationTargets {
  std::vector<int>    params;   // parameter indices to seed active
  std::vector<int>    ics;      // initial-state indices to seed active
  std::vector<double> values;   // seed values, params then ics

  bool empty() const { return params.empty() && ics.empty(); }
  std::size_t size() const { return params.size() + ics.size(); }
};

// Deactivates the tape on every exit from the driver below, including exceptions.
template <typename Tape>
struct tape_guard {
  Tape* tape;
  ~tape_guard() { tape->deactivate(); }
};

// Reverse-mode Jacobian of a functional of an ODE solve: row i is d(output_i)/d(input)
// over the seeded inputs. XAD's computeJacobian does the record-once, adjoint-row
// sweep; the forward callback here is the part XAD can't supply -- seed the registered
// inputs onto the System (ad_parameters / ad_initial_state), replay via solver.run(),
// and reduce the positioned solver through the functional.
//
// The replay is the solver's own run() -- advance_fixed over the schedule handed in via
// set_schedule() -- so the driver owns record/seed/sweep/reduce but never the schedule;
// the solver owns its replay.
//
// `codomain` is how many outputs the functional returns; XAD needs it to size the
// sweep, and reading it off the functional avoids a spare model run just to count.
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_jacobian(
    Solver& solver,
    const DifferentiationTargets& targets,
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
    // Every target must address an input the System actually declares.
    const std::size_t n_params = solver.get_system_ref().ad_parameters().size();
    const std::size_t n_ics    = solver.get_system_ref().ad_initial_state().size();
    for (int i : targets.params) {
        if (i < 0 || static_cast<std::size_t>(i) >= n_params)
            util::stop("DifferentiationTargets: parameter index out of range");
    }
    for (int j : targets.ics) {
        if (j < 0 || static_cast<std::size_t>(j) >= n_ics)
            util::stop("DifferentiationTargets: initial-state index out of range");
    }

    // The tape is created once and reused across the rows of this Jacobian.
    if (!solver.tape) {
        solver.tape = std::make_unique<ad::tape_type>(false);
    }
    solver.tape->activate();
    tape_guard<ad::tape_type> guard{solver.tape.get()};

    // computeJacobian registers the inputs, so it takes them as one flat vector
    // (ordered params-then-ics, matching `values`) and calls the forward callback.
    std::vector<ad_type> inputs(targets.values.begin(), targets.values.end());

    std::vector<double> values;
    std::function<std::vector<ad_type>(std::vector<ad_type>&)> forward =
        [&](std::vector<ad_type>& x) {
            auto& sys = solver.get_system_ref();
            auto params = sys.ad_parameters();
            auto ics    = sys.ad_initial_state();
            std::size_t k = 0;
            for (int i : targets.params) *params[i] = x[k++];
            for (int j : targets.ics)    *ics[j]    = x[k++];
            solver.reset();
            solver.run();
            auto outputs = functional(solver);
            values.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) values[i] = xad::value(outputs[i]);
            return outputs;
        };

    auto jacobian = xad::computeJacobian(inputs, forward, codomain, solver.tape.get());
    return {values, jacobian};
}

// Presents a scalar functional as a one-output functional for compute_jacobian.
template<typename Functional>
struct scalar_functional {
  Functional f;
  std::size_t codomain() const { return 1; }
  template<typename Solver>
  std::vector<typename Solver::value_type> operator()(Solver& s) const { return { f(s) }; }
};

// A gradient is the single row of the Jacobian of a scalar functional.
template<typename Solver, typename Functional>
std::pair<double, std::vector<double>> compute_gradient(
    Solver& solver,
    const DifferentiationTargets& targets,
    Functional&& functional
) {
    scalar_functional<std::decay_t<Functional>> one{std::forward<Functional>(functional)};
    auto [values, jacobian] = compute_jacobian(solver, targets, one);
    return {values[0], jacobian[0]};
}

// Forward-mode Jacobian-vector product: the directional derivative of a functional
// of an ODE solve along `direction` in the seeded-input space (`J v`). Forward mode
// needs NO tape -- one augmented run, the tangent travels alongside the value, so a
// growing-dimension resize is transparent (nothing to register, reserve, or keep
// alive across the resize -- all of which the reverse path must handle). `direction`
// is ordered params-then-ics, matching `values`. Returns {values, jvp} with
// jvp[i] = d(output_i)/d(direction).
//
// Its dual, `compute_gradient`/`compute_jacobian` (reverse), and this forward pass
// satisfy the adjoint dot-product identity <J v, u> = <v, Jᵀ u>: for random v, u the
// two must agree to machine precision, giving an FD-free correctness oracle (one
// forward + one reverse pass, no perturbation and no inner-solve re-run). It also
// wins outright when the codomain is small and the seeded-input set is large only in
// the *other* direction -- here it is the JVP primitive and the oracle's forward leg.
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<double>> compute_jvp(
    Solver& solver,
    const DifferentiationTargets& targets,
    const std::vector<double>& direction,
    Functional&& functional
) {
    using fwd_type = typename xad::fwd<double>::active_type;  // FReal<double>: one tangent layer

    if (targets.empty()) {
        util::stop("DifferentiationTargets must seed at least one input");
    }
    if (targets.size() != targets.values.size()) {
        util::stop("DifferentiationTargets: 'values' must match 'params' + 'ics'");
    }
    util::check_length(direction.size(), targets.size());
    const std::size_t n_params = solver.get_system_ref().ad_parameters().size();
    const std::size_t n_ics    = solver.get_system_ref().ad_initial_state().size();
    for (int i : targets.params) {
        if (i < 0 || static_cast<std::size_t>(i) >= n_params)
            util::stop("DifferentiationTargets: parameter index out of range");
    }
    for (int j : targets.ics) {
        if (j < 0 || static_cast<std::size_t>(j) >= n_ics)
            util::stop("DifferentiationTargets: initial-state index out of range");
    }

    // Seed the chosen inputs with their value and their tangent (the direction
    // component). No tape is created or activated -- forward mode is self-contained.
    auto& sys = solver.get_system_ref();
    auto params = sys.ad_parameters();
    auto ics    = sys.ad_initial_state();
    std::size_t k = 0;
    for (int i : targets.params) {
        *params[i] = fwd_type(targets.values[k]);
        xad::derivative(*params[i]) = direction[k];
        ++k;
    }
    for (int j : targets.ics) {
        *ics[j] = fwd_type(targets.values[k]);
        xad::derivative(*ics[j]) = direction[k];
        ++k;
    }
    solver.reset();
    solver.run();

    auto outputs = functional(solver);
    std::vector<double> values(outputs.size()), jvp(outputs.size());
    for (std::size_t o = 0; o < outputs.size(); ++o) {
        values[o] = xad::value(outputs[o]);
        jvp[o]    = xad::derivative(outputs[o]);
    }
    return {values, jvp};
}

// Scalar convenience: the directional derivative of a scalar functional along
// `direction`, returning {value, jvp}.
template<typename Solver, typename Functional>
std::pair<double, double> compute_directional_derivative(
    Solver& solver,
    const DifferentiationTargets& targets,
    const std::vector<double>& direction,
    Functional&& functional
) {
    scalar_functional<std::decay_t<Functional>> one{std::forward<Functional>(functional)};
    auto [values, jvp] = compute_jvp(solver, targets, direction, one);
    return {values[0], jvp[0]};
}

// Sum of squared residuals between predicted and measured observations.
template<typename T>
T sum_of_squares(const std::vector<std::vector<T>>& predicted,
                 const std::vector<std::vector<double>>& observations) {
    T loss(0.0);
    for (size_t i = 0; i < predicted.size(); ++i)
        for (size_t j = 0; j < predicted[i].size(); ++j) {
            T diff = predicted[i][j] - observations[i][j];
            loss += diff * diff;
        }
    return loss;
}

// One example functional: a least-squares calibration loss. It reads the replayed
// solver's collected states at the observation indices and scores them against
// measured data. It owns its data and samples the recorded grid, so the solver keeps
// no calibration state; calibration is just one functional among many.
struct least_squares {
  std::vector<size_t>              obs_indices;   // indices into the recorded steps
  std::vector<std::vector<double>> observations;  // measured data, per observation

  std::size_t codomain() const { return 1; }

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

} // namespace ode
} // namespace odelia

#endif
