#ifndef ODELIA_GRADIENT_HPP_
#define ODELIA_GRADIENT_HPP_

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/adjoint.hpp>
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

// A widening the run took, and the recorded step it follows. The index is taken
// where the widening happens rather than passed in, so it cannot disagree with
// the recording it indexes.
template <class Widening>
struct recorded_widening {
  std::size_t after_step;
  Widening event;
};

// A range of the recording over which the state has one width.
struct state_segment {
  std::size_t first, last;
};

// The recording cut into one range per width, lowest first. There is one more
// range than there are widenings, so the lowest runs down to the initial state
// whether or not a widening sits at the first recorded step.
//
// This is where a recording and the widenings claimed for it are checked against
// each other, and it is the check the shape exists for: the ranges have to
// partition the recording, or a sweep silently covers less of it than the
// trajectory it is transposing. Inferring the widenings from the state's width
// instead -- which is what a caller without a declared list must do -- cannot
// fail this test, because it defines the answer.
template <class Widening>
std::vector<state_segment> state_segments(
    const std::vector<recorded_widening<Widening>>& widenings,
    std::size_t n_recorded) {
    if (n_recorded < 2) {
        util::stop("state_segments: no recorded steps to sweep");
    }
    std::vector<state_segment> ret;
    ret.reserve(widenings.size() + 1);
    std::size_t first = 0;
    for (std::size_t j = 0; j < widenings.size(); ++j) {
        const std::size_t at = widenings[j].after_step;
        if (at + 1 >= n_recorded) {
            util::stop("state_segments: a widening follows a step the recording "
                       "does not have");
        }
        if (j > 0 && at <= widenings[j - 1].after_step) {
            util::stop("state_segments: the widenings are not in the order the "
                       "run took them");
        }
        ret.push_back({first, at});
        first = at;
    }
    ret.push_back({first, n_recorded - 1});
    return ret;
}

// Narrow the System through every widening, newest first, because each was
// applied on top of the one before it.
template <class System, class Widening>
  requires WidensState<System>
void narrow_all(System& system,
                const std::vector<recorded_widening<Widening>>& widenings,
                const std::vector<std::vector<double>>& states) {
    for (std::size_t j = widenings.size(); j-- > 0;) {
        system.narrow(widenings[j].event);
        util::check_length(system.ode_size(),
                           states[widenings[j].after_step].size());
    }
}

// And widen it back through all of them, which is the width the run left it at.
// Every walk here descends from that width and leaves the System lower, so each
// of them ends with this rather than leaving the next caller to notice.
//
// `applied` is how many widenings the System currently carries, because a walk
// that stopped at one segment carries that many and the way back starts by
// undoing exactly those. The caller knows the number; nothing here can read it
// off a width.
template <class System, class Widening>
  requires WidensState<System>
void widen_all(System& system,
               const std::vector<recorded_widening<Widening>>& widenings,
               const std::vector<std::vector<double>>& states,
               const std::vector<double>& times, std::size_t applied = 0) {
    if (applied > widenings.size()) {
        util::stop("widen_all: more widenings applied than the run took");
    }
    for (std::size_t j = applied; j-- > 0;) {
        system.narrow(widenings[j].event);
    }
    for (std::size_t j = 0; j < widenings.size(); ++j) {
        const std::size_t at = widenings[j].after_step;
        util::check_length(system.ode_size(), states[at].size());
        system.set_recorded_state(states[at].begin(), times[at]);
        system.widen(widenings[j].event);
    }
}

// The state a segment's first step ran from, and the time it ran at. It is what
// the run reached between a widening and the step after it, so no record holds
// it and it is replayed. The System is left at that segment's width, because a
// caller reading this state wants to run from it.
template <class System, class Widening>
  requires WidensState<System>
double state_at_segment(System& system,
                        const std::vector<recorded_widening<Widening>>& widenings,
                        const std::vector<std::vector<double>>& states,
                        const std::vector<double>& times, std::size_t segment,
                        std::vector<double>& base, std::size_t& start) {
    if (segment > widenings.size()) {
        util::stop("state_at_segment: the recording has no such segment");
    }
    system.set_recorded_state(states[0].begin(), times[0]);
    start = 0;
    for (std::size_t j = 0; j < segment; ++j) {
        const std::size_t at = widenings[j].after_step;
        system.set_recorded_state(states[at].begin(), times[at]);
        system.widen(widenings[j].event);
        start = at;
    }
    base.assign(system.ode_size(), 0.0);
    system.ode_state(base.begin());
    return times[start];
}

// Carry lambda back over a recording whose state widened, one segment per width,
// highest first. At the foot of every segment but the lowest sits a widening: the
// System is narrowed across it and the map that widened it is transposed there,
// so the rows it carries reach `parameter_adjoint` by the same route a step's do.
//
// `states[k]` is what the run held at times()[k]. The widened state between a
// widening and the step after it is what no record holds, so it is rebuilt here.
// `extra_splits` names steps at which a segment stops and resumes; a split
// outside every segment's interior cuts nothing. Returns how many ranges were
// swept, which is not the segment count -- an empty lowest segment is swept zero
// times, and a split adds one.
template <class Solver, class Widening>
  requires WidensState<typename Solver::system_type>
std::size_t solve_adjoint_over_widenings(
    Solver& solver, const std::vector<std::vector<double>>& states,
    const std::vector<recorded_widening<Widening>>& widenings,
    std::vector<std::vector<double>>& lambda,
    std::vector<std::vector<double>>& parameter_adjoint,
    const std::vector<std::size_t>& extra_splits = {}) {
    using scalar = active_scalar<double>;
    auto& system = solver.get_system_ref();
    const std::vector<state_segment> segments =
        state_segments(widenings, states.size());

    // narrow() has to be widen()'s inverse, and across several widenings it has
    // to be so in reverse order -- it drops what was added last, and nothing in
    // the signature says which that is. The state is the only witness: a wrong
    // entry dropped leaves the width right and the values wrong, which is the
    // shape every defect in this corpus takes.
    //
    // Widening and immediately narrowing the same event IS an exact identity,
    // so it costs one round trip per sweep to say so. Narrowing and widening
    // back is not, because what the run's own widening added has since moved.
    if (!widenings.empty()) {
        std::vector<double> before(system.ode_size());
        system.ode_state(before.begin());
        system.widen(widenings.back().event);
        system.narrow(widenings.back().event);
        std::vector<double> after(system.ode_size());
        if (after.size() != before.size()) {
            util::stop("solve_adjoint_over_widenings: narrow did not undo the "
                       "width widen added");
        }
        system.ode_state(after.begin());
        for (std::size_t i = 0; i < before.size(); ++i) {
            if (!util::identical(before[i], after[i])) {
                util::stop("solve_adjoint_over_widenings: narrow returned a "
                           "different state from the one widen was given, so it "
                           "is dropping something other than what widen added");
            }
        }
    }

    // The run left the System at its final width. Narrow to the lowest segment's,
    // newest widening first, because each narrowing drops what the one after it
    // was applied on top of.
    narrow_all(system, widenings, states);

    // Then back up, keeping the widened state each widening produced. A sweep
    // starting inside a segment runs from that state, and no record holds it.
    const std::vector<double> times = solver.times();
    std::vector<std::vector<double>> sweep_states = states;
    for (std::size_t j = 0; j < widenings.size(); ++j) {
        const std::size_t at = widenings[j].after_step;
        util::check_length(system.ode_size(), states[at].size());
        system.set_recorded_state(states[at].begin(), times[at]);
        system.widen(widenings[j].event);
        sweep_states[at].assign(system.ode_size(), 0.0);
        system.ode_state(sweep_states[at].begin());
        util::check_length(sweep_states[at].size(), states[at + 1].size());
    }

    // One tape for every widening this walk crosses. Clearing it returns it to an
    // empty recording and keeps the capacity, where one built per widening
    // regrows it.
    std::size_t swept = 0;
    typename scalar::tape_type tape(false);
    for (std::size_t j = segments.size(); j-- > 0;) {
        const state_segment& segment = segments[j];
        // Stopped and resumed at each requested step inside this segment, highest
        // first, so the pieces compose in the order the whole sweep would take.
        std::vector<std::size_t> cuts;
        for (const std::size_t s : extra_splits) {
            if (s > segment.first && s < segment.last) {
                cuts.push_back(s);
            }
        }
        std::sort(cuts.begin(), cuts.end());
        std::size_t upper = segment.last;
        for (std::size_t c = cuts.size(); c-- > 0;) {
            solver.solve_adjoint_batched(sweep_states, lambda,
                                         parameter_adjoint, cuts[c], upper);
            upper = cuts[c];
            ++swept;
        }
        // A widening at the first recorded step leaves the lowest segment with no
        // step in it, which is what a run from an empty state gives.
        if (segment.first < upper) {
            solver.solve_adjoint_batched(sweep_states, lambda,
                                         parameter_adjoint, segment.first,
                                         upper);
            ++swept;
        }
        if (j == 0) {
            break;
        }

        const recorded_widening<Widening>& w = widenings[j - 1];
        system.narrow(w.event);
        const double time = solver.times()[w.after_step];
        auto widen = [&](auto& active_system,
                         typename std::vector<scalar>::const_iterator x,
                         std::vector<scalar>& y) -> void {
            active_system.widened_state(w.event, time, x, y);
        };
        std::vector<std::vector<double>> narrowed;
        state_and_parameter_adjoints(tape, system, states[w.after_step], lambda,
                                     widen, narrowed, parameter_adjoint);
        lambda = std::move(narrowed);
    }
    return swept;
}

// Step `forward` over the recording's segments from `from_segment` on, at the
// sizes the run took, widening where the run widened. `first` is the recorded
// step the walk begins at, which the caller has already put the System on.
//
// The step sizes are replayed rather than the times: a size differenced back out
// of two recorded times is not the size that was taken, since fl(fl(t + h) - t)
// is not h, and a walk that chose its own would be differentiating a controller
// the model does not contain.
template <class Solver, class Widening>
  requires WidensState<typename Solver::system_type>
void advance_over_widenings(
    Solver& forward, const std::vector<recorded_widening<Widening>>& widenings,
    const std::vector<double>& step_sizes, std::size_t from_segment,
    std::size_t first) {
    const std::vector<state_segment> segments =
        state_segments(widenings, step_sizes.size());
    if (from_segment >= segments.size()) {
        util::stop("advance_over_widenings: the recording has no such segment");
    }
    for (std::size_t j = from_segment; j < segments.size(); ++j) {
        // The first entry is the size no step reached, which is how a recorded
        // run reads back.
        std::vector<double> sizes(1, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t k = first + 1; k <= segments[j].last; ++k) {
            sizes.push_back(step_sizes[k]);
        }
        if (sizes.size() > 1) {
            forward.advance_fixed_steps(sizes);
        }
        if (j + 1 < segments.size()) {
            forward.get_system_ref().widen(widenings[j].event);
            forward.set_state_from_system();
            first = segments[j].last;
        }
    }
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
