#ifndef ODELIA_ODE_SOLVER_HPP_
#define ODELIA_ODE_SOLVER_HPP_

#include <odelia/ode_solver_internal.hpp>
#include <XAD/XAD.hpp>
#include <XAD/Tape.hpp>
#include <cmath>
#include <memory>
#include <type_traits>

namespace odelia {
namespace ode {

// AD is opt-in: a System is differentiable only if it provides `rebind` (the double
// -> active mould). For a plain double System that never takes a gradient, resolve
// the active twin's type to the System itself -- a harmless placeholder that is
// declared but never constructed -- so `Solver<System>` compiles without forcing
// every System to carry AD scaffolding.
namespace detail {
template <class S, class Scalar, class = void>
struct rebind_or_self { using type = S; };
template <class S, class Scalar>
struct rebind_or_self<S, Scalar, std::void_t<typename S::template rebind<Scalar>>> {
  using type = typename S::template rebind<Scalar>;
};
}

// This is a wrapper class that is meant to simplify the
// difficuly of ownership semantics around the solver and system.
// It is mostly just be a generic wrapper around ode::Solver<System>
// You would only write your own if you had system-specific needs, e.g. events, cohort introudctions etc.

// TODO
// - add ability to set time directly
// - collect gathers vector of variables at each step
// - move collect into ode::Solver so that it can be used more generally
//   for any system, making this a generic Solver class

template <typename System>
class Solver
{
public:
  using value_type = typename System::value_type;

  Solver(System sys_, OdeControl control, Method method = Method::rkck)
    : system(sys_), control_(control), solver(system, control, method)
  {
    collect = true;
  }

  // Copyable: the tape and cached active twin are rebuildable amortization scratch
  // (RIF-3), not part of the Solver's value, so a copy starts with them empty and
  // rebuilds them lazily on its first gradient. plant copies Solvers on the non-AD
  // path (SCM snapshots, RcppR6 bindings), where that scratch is irrelevant; the
  // implicit copy ctor is deleted only because of the unique_ptr<Tape> member.
  Solver(const Solver& o)
    : collect(o.collect), system(o.system), control_(o.control_),
      solver(o.solver), replay_schedule_(o.replay_schedule_) {
    history = o.history;
  }
  Solver& operator=(const Solver& o) {
    collect = o.collect; system = o.system; control_ = o.control_;
    solver = o.solver; replay_schedule_ = o.replay_schedule_; history = o.history;
    tape.reset(); active_solver.reset(); adjoint_twin.reset();
    return *this;
  }
  Solver(Solver&&) = default;
  Solver& operator=(Solver&&) = default;

  // TODO: solver.reset() will set time within the solver to zero.
  // However, there is no other current way of setting the time within
  // the solver.  It might be better to add a set_time method within
  // ode::Solver, and then here do explicitly ode_solver.set_time(0)?
  void reset()
  {
    system.reset();
    solver.reset(system);
    history.clear();
  }

  // collectors
  double time() const { return solver.get_time(); }

  ode::state_type<System> state() const { return solver.get_state(); }
  std::vector<double> times() const { return solver.get_times(); }

  // The size of the step that reached each time in times(); NaN for the first,
  // which no step reached.
  std::vector<double> step_sizes() const { return solver.get_step_sizes(); }

  System get_system() const { return system; }
  System& get_system_ref() { return system; }

  // The control this solver was built with, so a driver builds the active solver
  // with the same integration settings.
  OdeControl get_control() const { return control_; }

  // Synchronize internal ODE buffers from the current system state without
  // resetting solver history/step-size state.
  void set_state_from_system()
  {
    solver.set_state_from_system(system);
  }

  void set_state(std::vector<double> y, double time)
  {
    util::check_length(y.size(), system.ode_size());
    internal::set_ode_state(system, y, time);
    solver.reset(system);
    solver.set_state_from_system(system);
  }

  // Take a series of adaptive steps up to some time
  void advance_adaptive(std::vector<double> times)
  {
    if (times.empty())
    {
      util::stop("'times' must be vector of at least length 1");
    }
    std::vector<double>::const_iterator t = times.begin();
    if (!util::identical(*t++, time()))
    {
      util::stop("First element in 'times' must be same as current time");
    }

    if (collect)
    {
      history.push_back(system);
    }

    while (t != times.end())
    {
      solver.advance_adaptive(system, *t++);
      if (collect)
      {
        history.push_back(system);
      }
    }
  }

  // Take a series of steps at specified time steps
  void advance_fixed(std::vector<double> times)
  {
    if (times.empty())
    {
      util::stop("'times' must be vector of at least length 1");
    }
    std::vector<double>::const_iterator t = times.begin();
    if (!util::identical(*t++, time()))
    {
      util::stop("First element in 'times' must be same as current time");
    }

    if (collect)
    {
      history.push_back(system);
    }

    while (t != times.end())
    {
      solver.step_to(system, *t++);
      if (collect)
      {
        history.push_back(system);
      }
    }
  }

  // Take a series of steps of the recorded sizes {NaN, h_1, h_2, ...}, in place of
  // differencing recorded times (cf. advance_fixed).
  void advance_fixed_steps(std::vector<double> step_sizes)
  {
    if (step_sizes.empty())
    {
      util::stop("'step_sizes' must be vector of at least length 1");
    }
    std::vector<double>::const_iterator h = step_sizes.begin();
    if (!std::isnan(*h++))
    {
      util::stop("First element in 'step_sizes' must be NaN, the recorded start");
    }

    if (collect)
    {
      history.push_back(system);
    }

    while (h != step_sizes.end())
    {
      solver.step_by(system, *h++);
      if (collect)
      {
        history.push_back(system);
      }
    }
  }

  // Take a series of plain forward-Euler steps over the supplied grid. One
  // derivative evaluation per step, no error control (cf. advance_fixed, which
  // drives the full RKCK stepper). Collects history at each supplied time.
  void advance_euler(std::vector<double> times)
  {
    if (times.empty())
    {
      util::stop("'times' must be vector of at least length 1");
    }
    std::vector<double>::const_iterator t = times.begin();
    if (!util::identical(*t++, time()))
    {
      util::stop("First element in 'times' must be same as current time");
    }

    if (collect)
    {
      history.push_back(system);
    }

    while (t != times.end())
    {
      solver.step_euler(system, *t++);
      if (collect)
      {
        history.push_back(system);
      }
    }
  }

  void step()
  {
    solver.step(system);
    if (collect)
    {
      history.push_back(system);
    }
  }

  bool get_collect() const { return collect; }

  void set_collect(bool x) { collect = x; }

  std::size_t get_history_size() const { return history.size(); }

  std::vector<System> get_history() const { return history; }

  System get_history_step(std::size_t i) const { return history.at(i); }

  // The read-only surface behind the "forgot to record" guard: whether an adaptive
  // pass has resolved a schedule on this solver, and what it is. The schedule is the
  // grid a replay-gradient advances over (advance_fixed).
  bool has_recording() const { return times().size() > 1; }
  std::vector<double> recorded_steps() const { return times(); }

  // Carry lambda back over the recorded steps, last to first. states[k] is the
  // state the run held at recorded_steps()[k], so step k ran from states[k - 1]
  // with step_sizes()[k]; on return lambda is the adjoint of states[0]. Only
  // accepted steps are recorded, and a rejected step never enters the solution,
  // so the recorded list is the whole of what the sweep visits.
  void solve_adjoint(const std::vector<ode::state_type<System> >& states,
                     ode::state_type<System>& lambda)
  {
    std::vector<double> no_parameters;
    solve_adjoint(states, lambda, no_parameters, 0, states.size() - 1);
  }

  // The same sweep restricted to steps k_last down to k_first + 1, so on return
  // lambda is the adjoint of states[k_first]. Every state it visits has to be
  // the width the System holds, so a caller whose System changes width between
  // steps sweeps one segment per width and changes the System in between.
  void solve_adjoint(const std::vector<ode::state_type<System> >& states,
                     ode::state_type<System>& lambda,
                     std::vector<double>& parameter_adjoint, size_t k_first,
                     size_t k_last)
  {
    if (!has_recording()) {
      util::stop("no recorded steps to sweep; run the adaptive pass first");
    }
    const std::vector<double> t = times();
    const std::vector<double> h = step_sizes();
    util::check_length(states.size(), t.size());
    util::check_length(lambda.size(), system.ode_size());
    if (k_first >= k_last || k_last >= t.size()) {
      util::stop("the adjoint segment is not a range of recorded steps");
    }
    ode::state_type<System> lambda_in(lambda.size());
    for (size_t k = k_last; k > k_first; --k) {
      util::check_length(states[k - 1].size(), system.ode_size());
      solver.step_adjoint(system, t[k - 1], h[k], states[k - 1], lambda,
                          lambda_in, parameter_adjoint);
      lambda = lambda_in;
    }
  }

  // The same segment, carrying one lambda per seed. Every seed sees the same
  // trajectory, so the stage states are rebuilt once per step however many are
  // carried and the System's transpose is recorded once -- which is where the
  // saving is, because a recording is a model evaluation and a sweep is not.
  void solve_adjoint_batched(const std::vector<ode::state_type<System> >& states,
                             std::vector<ode::state_type<System> >& lambda,
                             std::vector<std::vector<double>>& parameter_adjoint,
                             size_t k_first, size_t k_last)
  {
    if (!has_recording()) {
      util::stop("no recorded steps to sweep; run the adaptive pass first");
    }
    if (lambda.empty()) {
      util::stop("solve_adjoint_batched: needs at least one seed");
    }
    const std::vector<double> t = times();
    const std::vector<double> h = step_sizes();
    util::check_length(states.size(), t.size());
    for (const ode::state_type<System>& l : lambda) {
      util::check_length(l.size(), system.ode_size());
    }
    if (k_first >= k_last || k_last >= t.size()) {
      util::stop("the adjoint segment is not a range of recorded steps");
    }
    static_assert(Rebindable<System, active_scalar>,
                  "a batched adjoint sweep records on the System at the adjoint "
                  "scalar; this System has no rebind_from()");
    // One twin for the whole segment: its width is the width being swept, so a
    // segment at another width builds its own.
    if (!adjoint_twin || adjoint_twin->ode_size() != system.ode_size()) {
      adjoint_twin = std::make_unique<adjoint_twin_type>(
          system.template rebind_from<active_scalar>());
    }
    std::vector<ode::state_type<System> > lambda_in;
    for (size_t k = k_last; k > k_first; --k) {
      util::check_length(states[k - 1].size(), system.ode_size());
      solver.step_adjoint_batched(system, t[k - 1], h[k], states[k - 1], lambda,
                                  lambda_in, parameter_adjoint, *adjoint_twin);
      lambda = lambda_in;
    }
  }

  // Hand the recorded replay schedule (L1) to this solver. The active twin holds no
  // recording of its own (rebind copies values, not the schedule), so the schedule is
  // handed over per gradient call -- the L1 analogue of the System's set_recording for
  // L2/L3, and the reason L1 is Solver-owned state rather than a gradient-driver
  // argument.
  void set_schedule(std::vector<double> steps) { replay_schedule_ = std::move(steps); }

  // Replay the recorded schedule with the currently-seeded system: the forward pass
  // the gradient driver differentiates, called once per Jacobian row.
  void run() {
    if (replay_schedule_.empty()) {
      util::stop("no recorded schedule to replay; run the adaptive pass first");
    }
    advance_fixed(replay_schedule_);
  }

  // The active (AD) version of this System, lifted via rebind. Built on the first
  // gradient and reused, so an optimiser loop amortizes it (its own `tape` included).
  // Cached on the object rather than an R handle, so a C++ caller that holds the
  // solver as a plain member shares the reuse. mutable: scratch, reusable through a
  // const solver.
  using active_scalar      = ode::active_scalar<double>;
  using active_system_type = typename detail::rebind_or_self<System, active_scalar>::type;
  mutable std::shared_ptr<Solver<active_system_type>> active_solver;

  // Reverse-mode tape, created on the first gradient and reused (only ever exercised
  // on the active solver).
  std::unique_ptr<xad::Tape<double>> tape;

  // The System at the adjoint scalar the stage transposes record on, built on
  // the first batched sweep and handed to every stage of it. Rebuilt when the
  // System changes width; the System puts its own values into it per recording,
  // which is what a stage transpose is handed rather than a rebind of its own.
  using adjoint_twin_type =
      typename rebound_system<System, active_scalar,
                              Rebindable<System, active_scalar>>::type;
  mutable std::unique_ptr<adjoint_twin_type> adjoint_twin;

  // Should we record history at every step?
  // TODO: should this be part of ode_solver?
std::vector<System> history;

private:
  bool collect;
  System system;
  OdeControl control_;
  SolverInternal<System> solver;
  std::vector<double> replay_schedule_;  // L1 recording handed over per gradient call

};
}
}
#endif
