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

// Sum-of-squares loss between the trajectory and its targets. The default
// functional's reduction, kept separate so it can be reused and tested directly.
template<typename T>
T sum_of_squares(const std::vector<std::vector<T>>& obs,
                 const std::vector<std::vector<double>>& targets) {
    T loss(0.0);
    for (size_t i = 0; i < obs.size(); ++i) {
        for (size_t j = 0; j < obs[i].size(); ++j) {
            T diff = obs[i][j] - targets[i][j];
            loss += diff * diff;
        }
    }
    return loss;
}

// A "functional" is any callable that takes a seeded, freshly-reset Solver,
// drives it, and returns the scalar to differentiate. Driving is the
// functional's job, so odelia never learns what is being solved or scored --
// calibration here, an emergent metric in plant. This one, the default, scores
// the trajectory against its configured targets.
struct sum_of_squares_loss {
  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    return sum_of_squares(solver.advance_target(), solver.targets());
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

// Reverse-mode Jacobian of a multi-output functional -- `functional(solver)`
// returns the m outputs -- w.r.t. the n initial conditions and/or parameters.
// Delegates the record-once/row-sweep to xad::computeJacobian (the adjoint
// driver, optimal here: outputs << inputs, few metrics vs many traits). We
// supply the seam it doesn't know about: a `foo` that scatters the registered
// inputs into the System and drives the solver. Returns the m output values and
// the m x n Jacobian (row i = d(output_i)/d(input)).
//
// `codomain` is the number of outputs m. Passing it is not cosmetic: XAD runs
// `foo` an extra time just to size the output when it is left at 0 (the docs are
// explicit) -- one wasted forward eval, i.e. a full model replay when plant
// drives this. The caller knows m (the functional's codomain), so it threads it
// through.
template<typename Solver, typename Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_jacobian(
    Solver& solver,
    Functional&& functional,
    std::size_t codomain,
    std::optional<std::vector<double>> ic = std::nullopt,
    std::optional<std::vector<double>> params = std::nullopt
) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    if (!ic && !params) {
        util::stop("Must provide at least one of 'ic' or 'params'");
    }

    // Reuse tape across calls to avoid invalidating slots
    if (!solver.tape) {
        solver.tape = new ad::tape_type(false); // create without activating
    }
    solver.tape->activate();
    detail::tape_deactivate_guard<ad::tape_type> guard{solver.tape};

    // xad::computeJacobian owns the input registration, so it takes the inputs as
    // one flat vector and calls us back to map them onto the System; the plain
    // (non-registering) setters propagate those active values into the fields.
    std::vector<ad_type> inputs;
    if (params) inputs.insert(inputs.end(), params->begin(), params->end());
    if (ic)     inputs.insert(inputs.end(), ic->begin(), ic->end());
    const double t0 = solver.fit_times().empty() ? 0.0 : solver.fit_times()[0];

    std::vector<double> values;
    std::function<std::vector<ad_type>(std::vector<ad_type>&)> forward =
        [&](std::vector<ad_type>& x) {
            auto& system = solver.get_system_ref();
            auto it = x.begin();
            if (params) it = system.set_params(it);
            if (ic)     it = system.set_initial_state(it, t0);
            solver.reset();
            auto outputs = functional(solver);
            values.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                values[i] = xad::value(outputs[i]);
            }
            return outputs;
        };

    auto jacobian = xad::computeJacobian(inputs, forward, codomain, solver.tape);
    return {values, jacobian};
}

// Reverse-mode gradient of a scalar `functional(solver)` w.r.t. the initial
// conditions and/or parameters seeded active. A gradient is the one-row Jacobian
// of a scalar functional -- in adjoint mode with codomain = 1 the cost is
// identical (record once, one sweep) -- so this is a thin adapter over the one
// driver: wrap the scalar functional into a 1-vector functional and unwrap the
// single row. Keeping a single tape-management path means each System needs only
// the plain scatter setters, not the registering overloads compute_gradient used
// to require.
template<typename Solver, typename Functional>
std::pair<double, std::vector<double>> compute_gradient(
    Solver& solver,
    Functional&& functional,
    std::optional<std::vector<double>> ic = std::nullopt,
    std::optional<std::vector<double>> params = std::nullopt
) {
    using value_type = typename Solver::value_type;
    auto as_vector = [&](Solver& s) {
        return std::vector<value_type>{ functional(s) };
    };
    auto [values, jacobian] = compute_jacobian(solver, as_vector, 1, ic, params);
    return {values[0], jacobian[0]};
}

} // namespace ode
} // namespace odelia

#endif
