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

// Sum-of-squares between the model's predicted observations and the measured
// ones. The reduction of the least-squares functional, kept separate so it can
// be reused and tested directly. Up to the additive/scale constants of a
// homoscedastic Gaussian model this is the negative log-likelihood, so its
// minimiser is the maximum-likelihood fit.
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

// A "functional" is any callable that takes a seeded, freshly-reset Solver,
// drives it, and returns the scalar(s) to differentiate. odelia never learns
// what the scalar means -- a calibration loss, an emergent summary of the state
// (a stand's basal area), whatever the caller drives the solver to compute.
//
// `least_squares` is the one prebuilt calibration instance. Unlike an emergent
// functional it carries per-run data: the measured observations and the schedule
// at which to sample them. It owns that data itself -- the Solver is a generic
// trajectory sampler and stores no fit state -- so calibration is just another
// functional, not a special mode wired into the solver or the AD driver.
struct least_squares {
  std::vector<double>              times;         // schedule to advance through
  std::vector<size_t>              obs_indices;   // which time indices are observed
  std::vector<std::vector<double>> observations;  // measured data, per obs point

  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    return sum_of_squares(solver.advance_observations(times, obs_indices),
                          observations);
  }
};

namespace detail {
// Scope guard: deactivate the tape however we leave the driver -- normal return
// or *any* exception. Replaces the hand-written try/catch, which left `e`
// unused and, being `catch (const std::exception&)`, leaked an activated tape
// on a non-std throw.
template<typename Tape>
struct tape_deactivate_guard {
    Tape* tape;
    ~tape_deactivate_guard() { tape->deactivate(); }
};
} // namespace detail

// The active inputs a gradient is taken with respect to: leaf values, each
// addressed to a System field by a slot index. The System owns the slot->field
// routing (its scatter method); odelia only carries opaque (slot, value) pairs,
// so it never learns whether a leaf is a trait, an initial density, or a
// boundary flux. Any subset is expressible -- seed only ICs for IC sensitivity,
// only traits for calibration, or both -- with no per-kind branching. Column j
// of the Jacobian corresponds to (slots[j], values[j]).
struct Independents {
  std::vector<int>    slots;
  std::vector<double> values;   // parallel to slots

  bool empty() const { return values.empty(); }
};

// Reverse-mode Jacobian of a multi-output functional -- `functional(solver)`
// returns the m outputs -- w.r.t. the active inputs named by `independents`.
// Delegates the record-once/row-sweep to xad::computeJacobian (the adjoint
// driver, optimal here: outputs << inputs, few metrics vs many traits). We
// supply the seam it doesn't know about: the forward callback that scatters the
// registered inputs into the System and drives the solver. Returns the m output
// values and the m x n Jacobian (row i = d(output_i)/d(input)).
//
// `codomain` is the number of outputs m. Passing it is not cosmetic: XAD runs
// the forward callback an extra time just to size the output when it is left at
// 0 (the docs are explicit) -- one wasted forward eval, i.e. a full model replay
// when plant drives this. The caller knows m (the functional's codomain), so it
// threads it through.
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_jacobian(
    Solver& solver,
    const Independents& independents,
    Functional&& functional,
    std::size_t codomain
) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    if (independents.empty()) {
        util::stop("Independents must seed at least one leaf");
    }
    if (independents.slots.size() != independents.values.size()) {
        util::stop("Independents: 'slots' and 'values' must be the same length");
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
            auto outputs = functional(solver);
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
    const Independents& independents,
    Functional&& functional
) {
    using value_type = typename Solver::value_type;
    auto as_vector = [&](Solver& s) {
        return std::vector<value_type>{ functional(s) };
    };
    auto [values, jacobian] = compute_jacobian(solver, independents, as_vector, 1);
    return {values[0], jacobian[0]};
}

} // namespace ode
} // namespace odelia

#endif
