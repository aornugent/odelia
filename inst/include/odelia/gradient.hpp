#ifndef ODELIA_GRADIENT_HPP_
#define ODELIA_GRADIENT_HPP_

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_solver.hpp>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace odelia {
namespace ode {

// Forward-mode derivative of a one-input scalar function at a point. `f` is
// instantiated at the active scalar inside, so only doubles cross in and out.
//
// `f` must declare its return type. XAD's operators return expression templates
// holding references to their operands, so a deduced return type hands back
// references to temporaries that die on return, and reading the derivative then
// picks up reused stack memory. The static_assert rejects that shape.
template <typename F>
double forward_derivative(double x, F&& f) {
  using active_type = xad::fwd<double>::active_type;
  static_assert(
      std::is_same_v<std::invoke_result_t<F&, active_type&>, active_type>,
      "f must return the active scalar itself, not a deduced expression template");

  active_type x_active = x;
  xad::derivative(x_active) = 1.0;
  active_type y = f(x_active);
  return xad::derivative(y);
}

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

// The same block recorded ONCE and swept once per seed: `input_adjoints[m]` receives
// transpose(jacobian) * output_adjoints[m]. A caller wanting several rows of the same
// block pays one recording rather than one per row, and where the recording is the
// expensive part -- which is the case whenever `f` is a model evaluation rather than
// arithmetic -- that is the whole of the cost.
//
// Each sweep is bit-identical to the row a fresh recording of `f` would give, because
// clearDerivatives() returns the tape's derivative slots to zero while leaving the
// recorded operations alone. That is what makes this substitutable for the single-seed
// form above rather than an approximation of it.
//
// An empty seed is swept anyway rather than skipped: the row is then zeros, which is
// what the caller's accumulator expects, and skipping would make the result depend on
// which seeds happen to vanish at this state.
template <class F>
std::size_t vector_jacobian_products(xad::adj<double>::tape_type& tape,
                                     const std::vector<double>& x,
                                     const std::vector<std::vector<double>>& output_adjoints,
                                     F&& f,
                                     std::vector<std::vector<double>>& input_adjoints) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    ad::tape_type* active = ad::tape_type::getActive();
    if (active != nullptr && active != &tape) {
        util::stop("vector_jacobian_products: a tape is already active");
    }
    if (x.empty()) {
        util::stop("vector_jacobian_products: 'x' must have at least one entry");
    }
    if (output_adjoints.empty()) {
        util::stop("vector_jacobian_products: needs at least one seed");
    }
    const std::size_t n_out = output_adjoints.front().size();
    if (n_out == 0) {
        util::stop("vector_jacobian_products: a seed must have at least one entry");
    }
    for (const std::vector<double>& s : output_adjoints) {
        if (s.size() != n_out) {
            util::stop("vector_jacobian_products: every seed must have one entry per "
                       "output; they are swept over one recording of the same block");
        }
    }

    tape.activate();
    tape_guard<ad::tape_type> guard{&tape};
    tape.clearAll();

    std::vector<ad_type> x_active(x.begin(), x.end());
    tape.registerInputs(x_active);
    tape.newRecording();

    std::vector<ad_type> y_active(n_out);
    f(x_active, y_active);
    if (y_active.size() != n_out) {
        util::stop("vector_jacobian_products: 'f' resized the output buffer; it is "
                   "handed one entry per output adjoint and must write in place");
    }
    tape.registerOutputs(y_active);

    input_adjoints.resize(output_adjoints.size());
    for (std::size_t m = 0; m < output_adjoints.size(); ++m) {
        // Between sweeps, or the previous seed's adjoints are still on the slots and
        // every row after the first is the running sum of the ones before it.
        tape.clearDerivatives();
        for (std::size_t i = 0; i < y_active.size(); ++i) {
            xad::derivative(y_active[i]) = output_adjoints[m][i];
        }
        tape.computeAdjoints();
        input_adjoints[m].resize(x.size());
        for (std::size_t i = 0; i < x_active.size(); ++i) {
            input_adjoints[m][i] = xad::derivative(x_active[i]);
        }
    }

    return tape.getMemory();
}

// A transpose taken with respect to a System's state AND the parameters its
// active twin lists: one recording over both, swept once per seed.
//
// `evaluate` is handed the state half of the recorded inputs and the output
// buffer, with the twin's parameters ALREADY written from the other half. That
// order is why the two halves are one recording rather than two calls: a
// quantity the state determines reads the parameters while deriving it, so a
// state written first derives it at the previous values. Writing the parameters
// here rather than in `evaluate` is what keeps that from being a rule each
// caller has to remember.
//
// The sweep splits back along the same seam, and the two halves are handled
// differently. `state_adjoint` is REPLACED: it is resized to one row per seed and
// each row assigned, so rows a shorter batch does not reach do not survive as
// stale numbers. `parameter_adjoint` is ADDED to, because a parameter is reached
// once per step and its gradient is the sum over every step swept, so it is the
// caller's to clear once per sweep and may carry rows past the seeds handed in.
//
// Those two are therefore different objects, and one being the other would mean
// resizing the accumulator between the check on its rows and the writes into
// them. Refused rather than documented.
//
// `twin` must be rebound for this call and not carried over from a previous one.
// Its parameters are written from the recorded inputs, so a twin that was written
// that way before arrives holding scalars from a recording this one has cleared,
// and the sweep comes back wrong -- not obviously wrong: one seed's rows can stay
// exact while another's do not. Nothing here can see the difference, which is why
// it is stated. A caller wanting to amortise the rebind has to make the twin's
// scalars fresh some other way.
template <class Twin, class Evaluate>
std::size_t state_and_parameter_adjoints(
    xad::adj<double>::tape_type& tape, Twin& twin,
    const std::vector<double>& state,
    const std::vector<std::vector<double>>& output_adjoints, Evaluate&& evaluate,
    std::vector<std::vector<double>>& state_adjoint,
    std::vector<std::vector<double>>& parameter_adjoint) {
    using scalar = active_scalar<double>;
    const std::vector<scalar*> parameters = twin.ad_parameters();
    const std::size_t n_state = state.size();
    const std::size_t n_parameter = parameters.size();
    const std::size_t n_seed = output_adjoints.size();

    if (&state_adjoint == &parameter_adjoint) {
        util::stop("state_and_parameter_adjoints: the state adjoints are "
                   "replaced and the parameter adjoints accumulated, so they "
                   "cannot be the same vector");
    }
    if (parameter_adjoint.size() < n_seed) {
        util::stop("state_and_parameter_adjoints: one row of parameter adjoints "
                   "per seed, to accumulate into");
    }
    for (std::size_t m = 0; m < n_seed; ++m) {
        util::check_length(parameter_adjoint[m].size(), n_parameter);
    }

    std::vector<double> in(state);
    in.reserve(n_state + n_parameter);
    for (const scalar* p : parameters) {
        in.push_back(util::to_passive(*p));
    }

    auto record = [&](const std::vector<scalar>& x,
                      std::vector<scalar>& y) -> void {
        std::size_t at = n_state;
        for (scalar* p : parameters) {
            *p = x[at++];
        }
        evaluate(x.begin(), y);
    };

    std::vector<std::vector<double>> in_adjoint;
    const std::size_t recording =
        vector_jacobian_products(tape, in, output_adjoints, record, in_adjoint);

    state_adjoint.assign(n_seed, std::vector<double>(n_state, 0.0));
    for (std::size_t m = 0; m < n_seed; ++m) {
        for (std::size_t j = 0; j < n_state; ++j) {
            state_adjoint[m][j] = in_adjoint[m][j];
        }
        for (std::size_t p = 0; p < n_parameter; ++p) {
            parameter_adjoint[m][p] += in_adjoint[m][n_state + p];
        }
    }
    return recording;
}

// One block of `f`, recorded and swept once on the tape handed in: `input_adjoints`
// receives transpose(jacobian) * output_adjoints, and the return value is the recording's
// size. `f` is instantiated at the active scalar here, so only doubles cross in and out.
//
// The tape is the caller's and is reused across calls, so nothing here allocates one; a
// tape costs about a fifth of this whole product and the product runs millions of times
// per gradient.
//
// Stops if a tape other than this one is active. Recording onto a tape this product does
// not own would sweep the block's adjoints twice, so "the tape handed in is the only one"
// is checked rather than assumed.
template <class F>
std::size_t vector_jacobian_product(xad::adj<double>::tape_type& tape,
                                    const std::vector<double>& x,
                                    const std::vector<double>& output_adjoints, F&& f,
                                    std::vector<double>& input_adjoints) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    ad::tape_type* active = ad::tape_type::getActive();
    if (active != nullptr && active != &tape) {
        util::stop("vector_jacobian_product: a tape is already active");
    }
    if (x.empty()) {
        util::stop("vector_jacobian_product: 'x' must have at least one entry");
    }
    if (output_adjoints.empty()) {
        util::stop("vector_jacobian_product: 'output_adjoints' must have at least one entry");
    }

    tape.activate();
    tape_guard<ad::tape_type> guard{&tape};

    // clearAll() returns the tape to an empty recording with its derivative-slot counter
    // back at zero. newRecording() alone leaves that counter where the previous call left
    // it -- destroying a registered input only releases its slot when the slot is the last
    // one, which it is not for a vector destroyed front to back -- so the tape's memory and
    // variable count would climb with every call while the adjoints stayed correct.
    tape.clearAll();

    // Inputs are registered before newRecording(). Registering after it leaves them outside
    // the recording, and the sweep then reports every input adjoint as zero with nothing
    // thrown.
    std::vector<ad_type> x_active(x.begin(), x.end());
    tape.registerInputs(x_active);
    tape.newRecording();

    std::vector<ad_type> y_active(output_adjoints.size());
    f(x_active, y_active);
    if (y_active.size() != output_adjoints.size()) {
        util::stop("vector_jacobian_product: 'f' resized the output buffer; it is "
                   "handed one entry per output adjoint and must write in place");
    }
    tape.registerOutputs(y_active);

    for (std::size_t i = 0; i < y_active.size(); ++i) {
        xad::derivative(y_active[i]) = output_adjoints[i];
    }
    tape.computeAdjoints();

    // The caller owns the buffer and reuses it across calls, so this resize is a no-op
    // after the first call and the product never allocates its own result.
    input_adjoints.resize(x.size());
    for (std::size_t i = 0; i < x_active.size(); ++i) {
        input_adjoints[i] = xad::derivative(x_active[i]);
    }

    return tape.getMemory();
}

// The same product for a caller with no tape to reuse: one call, one tape. Constructed
// inactive, so a tape already active here belongs to someone else and the overload above
// stops on it.
template <class F>
std::size_t vector_jacobian_product(const std::vector<double>& x,
                                    const std::vector<double>& output_adjoints, F&& f,
                                    std::vector<double>& input_adjoints) {
    xad::adj<double>::tape_type tape(false);
    return vector_jacobian_product(tape, x, output_adjoints, std::forward<F>(f), input_adjoints);
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
