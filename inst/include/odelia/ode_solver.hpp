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

  // Copyable: the tape and the cached active solver are rebuildable scratch, not
  // part of the Solver's value, so a copy starts with them empty and rebuilds them
  // on its first gradient. plant copies Solvers on the non-AD path, where that
  // scratch is irrelevant; the implicit copy constructor is unavailable only
  // because of the unique_ptr<Tape> member.
  Solver(const Solver& o)
    : collect(o.collect), system(o.system), control_(o.control_),
      solver(o.solver), replay_schedule_(o.replay_schedule_) {
    history = o.history;
  }
  Solver& operator=(const Solver& o) {
    collect = o.collect; system = o.system; control_ = o.control_;
    solver = o.solver; replay_schedule_ = o.replay_schedule_; history = o.history;
    tape.reset(); active_solver.reset();
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

  // The System this solver steps, so a caller taking the solver can constrain on
  // it rather than on what get_system_ref() happens to return.
  using system_type = System;

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

  // Keep the state at each accepted step beside the time and the size that reached
  // it. Set before the run, because the state the run starts from is recorded as it
  // begins.
  void set_keep_states(bool keep) { solver.set_keep_states(keep); }
  bool keeps_states() const { return solver.keeps_states(); }
  const ode::state_type<System>& recorded_state(std::size_t k) const {
    return solver.recorded_state(k);
  }
  std::size_t recorded_steps() const { return solver.recorded_steps(); }

  // The read-only surface behind the "forgot to record" guard: whether an adaptive
  // pass has resolved a schedule on this solver, and what it is. The schedule is the
  // grid a replay-gradient advances over (advance_fixed).
  bool has_recording() const { return times().size() > 1; }

  // Carry lambda back over recorded steps k_last down to k_first + 1, so on return
  // lambda is the adjoint of states[k_first]. states[k] is the state the run held
  // at times()[k], so step k ran from states[k - 1] with step_sizes()[k]. Only
  // accepted steps are recorded, and a rejected step never enters the solution, so
  // the recorded list is the whole of what a sweep visits.
  //
  // Every state visited has to be the width the System holds, so a caller whose
  // System changes width between steps sweeps one segment per width and changes the
  // System in between.
  //
  // One lambda per seed, and every seed sees the same trajectory -- so each step is
  // recorded once however many are carried, and swept per seed. That is where the
  // saving is, because a recording is a model evaluation and a sweep is not. A
  // caller wanting a single row passes a batch of one.
  void solve_adjoint(const std::vector<ode::state_type<System> >& states,
                     std::vector<ode::state_type<System> >& lambda,
                     std::vector<std::vector<double>>& parameter_adjoint,
                     size_t k_first, size_t k_last)
  {
    if (!has_recording()) {
      util::stop("no recorded steps to sweep; run the adaptive pass first");
    }
    if (lambda.empty()) {
      util::stop("solve_adjoint: needs at least one seed");
    }
    const std::vector<double> t = times();
    const std::vector<double> h = step_sizes();
    util::check_length(states.size(), t.size());
    for (const ode::state_type<System>& l : lambda) {
      // Named, because a bare length mismatch here is read as the caller's and says
      // nothing about the seam it is really about: a seed carried at one width
      // against a System left at another.
      if (l.size() != system.ode_size()) {
        util::stop("solve_adjoint: a seed is " +
                   util::to_string(static_cast<int>(l.size())) +
                   " wide against a System of " +
                   util::to_string(static_cast<int>(system.ode_size())) +
                   ", so the two are not at the same widening");
      }
    }
    if (k_first >= k_last || k_last >= t.size()) {
      util::stop("the adjoint segment is not a range of recorded steps");
    }
    std::vector<ode::state_type<System> > lambda_in;
    for (size_t k = k_last; k > k_first; --k) {
      util::check_length(states[k - 1].size(), system.ode_size());
      solver.step_adjoint(system, k, t[k - 1], h[k], states[k - 1], lambda,
                          lambda_in, parameter_adjoint);
      lambda = lambda_in;
    }
  }

  // Rate evaluations the sweeps since the last clear have recorded: six a step,
  // whatever the seed count. A term entering once a step where it belongs once a
  // stage divides this by six, and no gradient check can see that, because a
  // tangent and a sweep apply the same multiplier.
  std::size_t recorded_rates() const { return solver.recorded_rates(); }
  void clear_recorded_rates() { solver.clear_recorded_rates(); }

  // Hand the recorded replay schedule (L1) to this solver. The active System holds no
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
  using active_system_type = typename rebound_system<System, active_scalar>::type;
  mutable std::shared_ptr<Solver<active_system_type>> active_solver;

  // Reverse-mode tape, created on the first gradient and reused (only ever exercised
  // on the active solver).
  std::unique_ptr<xad::Tape<double>> tape;

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
