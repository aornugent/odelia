#ifndef ODELIA_ODE_SOLVER_HPP_
#define ODELIA_ODE_SOLVER_HPP_

#include <odelia/ode_solver_internal.hpp>
#include <XAD/XAD.hpp>
#include <XAD/Tape.hpp>
#include <memory>

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

  Solver(System sys_, OdeControl control)
    : system(sys_), control_(control), solver(system, control)
  {
    collect = true;
  }

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

  System get_system() const { return system; }
  System& get_system_ref() { return system; }

  // The control this solver was built with -- lets a driver construct the active
  // replay (RIF-2 rebind) with the same integration settings (RIF-1).
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

  // Advance through `times`, sampling the model state at the scheduled
  // observation indices. A generic trajectory-sampling primitive: it holds no
  // calibration data and knows nothing of "targets" or a loss. A functional that
  // scores the run against measured data (ode::least_squares) owns the
  // observations and passes the schedule in; any other functional ignores this
  // and reads the solved state its own way. Returns the model's predicted
  // observations -- state() at each time index named in `obs_indices`.
  std::vector<std::vector<typename System::value_type>> advance_observations(
      const std::vector<double>& times,
      const std::vector<size_t>& obs_indices) {
    if (times.empty()) {
      util::stop("advance_observations: 'times' must not be empty");
    }
    if (!util::identical(times[0], time())) {
      util::stop("First element in times must be same as current time");
    }

    std::vector<std::vector<typename System::value_type>> observations;
    observations.reserve(obs_indices.size());

    // Track which observation we're looking for
    size_t obs_idx = 0;

    // Check if initial time is an observation
    if (obs_idx < obs_indices.size() && obs_indices[obs_idx] == 0) {
      observations.push_back(state());
      obs_idx++;
    }

    // Step through times
    for (size_t i = 1; i < times.size(); ++i) {
      solver.step_to(system, times[i]);

      // Check if this time index is an observation point
      while (obs_idx < obs_indices.size() && obs_indices[obs_idx] == i) {
        observations.push_back(state());
        obs_idx++;
      }
    }

    return observations;
  }

  // The honest surface behind the "forgot to record" guard (ad-r-interface.md
  // §6.7): whether an adaptive pass has resolved a schedule on this (double) solver,
  // and what that schedule is. The schedule is the L1 recording every solver carries
  // -- a replay-gradient reads it (as the advance_fixed grid) and can check it first.
  bool has_recording() const { return times().size() > 1; }
  std::vector<double> recorded_steps() const { return times(); }

  // ---- AD scratch, reused across gradient calls (RIF-3) ----------------------
  // Both are lazily built on the first gradient call and reused thereafter, so an
  // optimiser loop amortizes them instead of reallocating each iteration; both are
  // null for a solver that never takes a gradient (nothing forces a System to be
  // differentiable). They live on the solver OBJECT -- not an R handle -- so a C++
  // caller that owns the solver as a plain member (plant's SCM) shares the reuse.
  // Reusing them is pure speed: it never changes a number. The recording, read per
  // call from the immutable double System, is what carries semantics.
  //
  //   active_solver -- this System lifted to the active scalar (RIF-2 rebind): the
  //     differentiable solver the gradient actually runs on. Its type is named via
  //     the System's own `rebind`, so the cache needs no void* / static_cast. Its own
  //     `tape` is the reused tape -- there is nothing else to cache.
  // mutable: incidental scratch, reusable through an otherwise-const solver.
  using active_system_type =
      typename System::template rebind<typename xad::adj<double>::active_type>;
  mutable std::shared_ptr<Solver<active_system_type>> active_solver;

  // Reverse-mode tape, created on the first gradient and reused (only ever exercised
  // on the active solver). unique_ptr so ownership is self-evident -- no destructor.
  std::unique_ptr<xad::Tape<double>> tape;

  // Should we record history at every step?
  // TODO: should this be part of ode_solver?
std::vector<System> history;

private:
  bool collect;
  System system;
  OdeControl control_;
  SolverInternal<System> solver;

};
}
}
#endif
