// -*-c++-*-
#ifndef ODELIA_ODE_SOLVER_INTERNAL_HPP_
#define ODELIA_ODE_SOLVER_INTERNAL_HPP_

#include <odelia/ode_interface.hpp>
#include <odelia/ode_control.hpp>
#include <odelia/ode_step.hpp>
#include <odelia/ode_step_rodas.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>

namespace odelia {
namespace ode {

// Integration method: the explicit Cash-Karp RKCK 4(5) stepper (default) or the
// implicit RODAS4(3) Rosenbrock stepper for stiff systems.
enum class Method { rkck, rodas };

template <class System>
class SolverInternal {
public:
  // Extract scalar type from System using traits
  using value_type = typename System::value_type;
  using state_type = std::vector<value_type>;

  // The system is taken by mutable reference throughout: reading its rates may
  // require it to compute them for the state it currently holds. See
  // set_state_from_system.
  SolverInternal(System &system, OdeControl control_,
                 Method method_ = Method::rkck);
  void reset(System& system);
  void set_state_from_system(System& system);

  state_type get_state() const {return y;}
  double get_time() const {return time;}
  std::vector<double> get_times() const;
  std::vector<double> get_step_sizes() const;

  void advance_adaptive(System &system, double time_max_);
  void advance_fixed(System& system, const std::vector<double>& times);
  void advance_fixed_steps(System& system,
                           const std::vector<double>& step_sizes);
  void advance_euler(System& system, const std::vector<double>& times);

  void step(System& system);

  // The adjoint of one step, from the state that step started at, for several
  // seeds at once. RKCK only: the Rosenbrock stepper carries no reverse
  // counterpart.
  void step_adjoint(System& system, std::size_t step, double time,
                    double step_size, const state_type& y,
                    const std::vector<state_type>& lambda_out,
                    std::vector<state_type>& lambda_in,
                    std::vector<std::vector<double>>& parameter_adjoint) {
    if (method == Method::rodas) {
      util::stop("method='rodas' has no adjoint; use method='rkck'.");
    }
    // The System can be a different width from the one the forward pass left,
    // because a caller sweeping a segment narrows it between segments. Every seed
    // carries that same width, and the stage buffers are sized to it here. A sweep
    // is the end of the solver's forward state either way.
    resize(lambda_out.front().size());
    stepper.step_adjoint(system, step, time, step_size, y, lambda_out,
                         lambda_in, parameter_adjoint);
  }

  // Rate evaluations recorded since the count was last cleared.
  std::size_t recorded_rates() const { return stepper.recorded_rates; }
  void clear_recorded_rates() { stepper.recorded_rates = 0; }


  // Keep the state at each accepted step as well as the time and the size. The
  // caller's decision: a run whose gradient will be taken needs the states, and a
  // run that is only integrating does not.
  void set_keep_states(bool keep) { keep_states_ = keep; }
  bool keeps_states() const { return keep_states_; }
  // What the run held at accepted step k, out of the record it shares with the
  // time and the size that reached it -- so a caller cannot pair one run's state
  // with another run's size.
  const state_type& recorded_state(std::size_t k) const {
    return prev_steps.at(k).state;
  }
  std::size_t recorded_steps() const { return prev_steps.size(); }

  // One accepted step, into the record: the time it reached, the size that
  // reached it, and the state there where the run was asked to keep states.
  void push_step(System& system, double time_, double step_size);
  void step_to(System& system, double time_max_);
  void step_by(System& system, double step_size);
  void step_euler(System& system, double time_max_);

  void set_time_max(double time_max_);

private:
  void resize(size_t size_);
  void setup_dydt_in(System& system);
  void save_dydt_out_as_in();
  void set_time(double t);

  // Stepper dispatch: SolverInternal holds both steppers and forwards to the one
  // selected at construction. The adaptive controller (see step()) is otherwise
  // stepper-agnostic.
  // The step index the stages are addressed by is this object's own count of
  // accepted steps, which is the step about to be taken -- read here rather than
  // passed, so no caller can disagree with it.
  void stepper_step(System& system, double time_, double step_size,
                    state_type& y_, state_type& yerr_,
                    const state_type& dydt_in_, state_type& dydt_out_) {
    if (method == Method::rodas) {
      if constexpr (RodasStep<System>::supported) {
        rodas_stepper.step(system, time_, step_size, y_, yerr_, dydt_in_,
                           dydt_out_);
      } else {
        // RODAS is unavailable for this System: either it provides no rebind_from()
        // hook for the AD Jacobian, or its scalar type is itself active (nested
        // tangent-over-adjoint is not yet wired up -- see issue #35). The passive
        // solver of a system with rebind_from() supports RODAS.
        util::stop("method='rodas' is not available for this system/scalar type "
                   "(needs a rebind_from() hook and a non-active scalar); "
                   "use method='rkck'.");
      }
    } else {
      stepper.step(system, prev_steps.size(), time_, step_size, y_, yerr_,
                   dydt_in_, dydt_out_);
    }
  }
  size_t stepper_order() const {
    return method == Method::rodas ? rodas_stepper.order() : stepper.order();
  }
  bool stepper_can_use_dydt_in() const {
    return method == Method::rodas ? RodasStep<System>::can_use_dydt_in
                                   : Step<System>::can_use_dydt_in;
  }
  bool stepper_first_same_as_last() const {
    return method == Method::rodas ? RodasStep<System>::first_same_as_last
                                   : Step<System>::first_same_as_last;
  }

  OdeControl control;
  Method method;
  Step<System> stepper;
  RodasStep<System> rodas_stepper;

  double step_size_last; // Size of last successful step (or suggestion)

  double time;     // Current time
  double time_max; // Time we will not go past
  // Each accepted step: the time it reached and the size it took. The size is
  // recorded rather than recovered from successive times because
  // fl(fl(t + h) - t) != h -- the addition rounds away bits of h that the
  // subtraction cannot return, so a replay that differences the times takes
  // different steps from the run it replays.
  // One entry per accepted step, the first being the state the run started from,
  // which no step reached and which therefore has no size.
  //
  // The state is kept BESIDE the size that reached it because the two are one
  // record. Where they live in separate stores a walk can pair a state with a size
  // from a different run, and nothing says so -- so the store that held the states
  // had to be emptied as it was read, and every consumer after the first repeated
  // the whole run to refill it.
  struct step_record {
    double time;
    double step_size;
    state_type state;   // empty unless this run was asked to keep them
  };
  std::vector<step_record> prev_steps;
  // Whether to keep the states. The caller's: a run whose gradient will be taken
  // needs them and a run that is only integrating does not.
  bool keep_states_ = false;

  state_type y;        // Vector of current system state
  state_type yerr;     // Vector of error estimates
  state_type dydt_in;  // Vector of dydt at beginning of step
  state_type dydt_out; // Vector of dydt during step

  bool dydt_in_is_clean;
};

// NOTE I'm setting the initial system size to 0 here, but some
// systems are self-initialising.
template <class System>
SolverInternal<System>::SolverInternal(System &system, OdeControl control_,
                                       Method method_)
  : control(control_), method(method_) {
  reset(system);
}

// NOTE: This resets *everything* to basically a recreated object.
template <class System>
void SolverInternal<System>::reset(System& system) {
  prev_steps.clear();
  step_size_last = control.step_size_initial;
  time_max = std::numeric_limits<double>::infinity();
  set_state_from_system(system);
}

// Seed y and dydt_in from whatever state the system currently holds. The system
// is mutable because `ode_rates` is allowed to compute: a system that reaches a
// state by a route of its own (widening it, reloading it) can then hand back the
// derivative *of that state* rather than a cached one belonging to an earlier
// one. Marking dydt_in clean here is only sound because of that -- with a const
// system the rates were whatever the system last happened to store, and under
// first-same-as-last they became k1 of the next step.
template <class System>
void SolverInternal<System>::set_state_from_system(System& system) {
  set_time(ode::ode_time(system));
  resize(system.ode_size());
  system.ode_state(y.begin());
  system.ode_rates(dydt_in.begin());
  dydt_in_is_clean = true;
  // The state the run starts from, into the record set_time just opened. Kept
  // here rather than by the caller because this is the one place that knows the
  // System has been read.
  if (keep_states_ && prev_steps.size() == 1 && prev_steps.back().state.empty()) {
    prev_steps.back().state.assign(y.begin(), y.end());
  }
}

// One accepted step, recorded. The state comes off the System rather than out of
// `y`, so a System that reaches a state by a route of its own is recorded at the
// state it holds.
template <class System>
void SolverInternal<System>::push_step(System& system, double time_,
                                       double step_size) {
  step_record record{time_, step_size, state_type()};
  if (keep_states_) {
    record.state.resize(system.ode_size());
    system.ode_state(record.state.begin());
  }
  prev_steps.push_back(std::move(record));
}

template <class System>
std::vector<double> SolverInternal<System>::get_times() const {
  std::vector<double> ret;
  ret.reserve(prev_steps.size());
  for (const step_record& s : prev_steps) {
    ret.push_back(s.time);
  }
  return ret;
}

// The size of the step that reached each recorded time; NaN for the initial
// time, which no step reached.
template <class System>
std::vector<double> SolverInternal<System>::get_step_sizes() const {
  std::vector<double> ret;
  ret.reserve(prev_steps.size());
  for (const step_record& s : prev_steps) {
    ret.push_back(s.step_size);
  }
  return ret;
}

template <class System>
void SolverInternal<System>::advance_adaptive(System &system, double time_max_)
{
  set_time_max(time_max_);
  while (time < time_max) {
    step(system);
  }
}

// NOTE: We take a vector of times {t_0, t_1, ...}.  This vector
// *must* contain a starting time, but can otherwise be empty.  We
// will step exactly to t_1, then to t_2 up to the end point.  No step
// size adjustments will be done.  This is used in the SCM.
//
// NOTE: Careful here: exact floating point comparison in determining
// that we're starting from the right place.  However, because we take
// care to return and add end points exactly, this should actually be
// the correct move.
template <class System>
void SolverInternal<System>::advance_fixed(System& system,
                                   const std::vector<double>& times) {
  if (times.empty()) {
    util::stop("'times' must be vector of at least length 1");
  }
  std::vector<double>::const_iterator t = times.begin();
  if (!util::identical(*t++, time))
  {
    util::stop("First element in 'times' must be same as current time");
  }
  while (t != times.end()) {
    step_to(system, *t++);
  }
}

// Step over a recorded run's step sizes {NaN, h_1, h_2, ...}, one step each, in
// place of differencing its times. The leading NaN is the recorded start, which
// no step reached; requiring it here is what checks that the caller handed over a
// whole recording rather than a sequence offset by one step.
template <class System>
void SolverInternal<System>::advance_fixed_steps(System& system,
                                   const std::vector<double>& step_sizes) {
  if (step_sizes.empty()) {
    util::stop("'step_sizes' must be vector of at least length 1");
  }
  std::vector<double>::const_iterator h = step_sizes.begin();
  if (!std::isnan(*h++)) {
    util::stop("First element in 'step_sizes' must be NaN, the recorded start");
  }
  while (h != step_sizes.end()) {
    step_by(system, *h++);
  }
}

// Plain forward (explicit) Euler integration over a supplied grid
// {t_0, t_1, ...}.  Unlike advance_fixed (which still drives the full multi-stage
// RKCK stepper at each interval), this does ONE derivative evaluation per
// interval: derivatives at the current state, then y <- y + h * dydt, advancing
// the time exactly to each grid point.  The `Step` (RKCK) machinery is bypassed
// entirely, so there is no error estimate and no step-size control.  Used to run
// systems the way fixed-step DGVMs do (e.g. a daily step).
template <class System>
void SolverInternal<System>::advance_euler(System& system,
                                   const std::vector<double>& times) {
  if (times.empty()) {
    util::stop("'times' must be vector of at least length 1");
  }
  std::vector<double>::const_iterator t = times.begin();
  if (!util::identical(*t++, time)) {
    util::stop("First element in 'times' must be same as current time");
  }
  while (t != times.end()) {
    step_euler(system, *t++);
  }
}

// A single forward-Euler step from the current time up to time_max_.  One
// derivative evaluation; no error estimate.  Leaves the system synchronised with
// the new state (like step_to, whose final RK derivs settles the system at y) so
// that collected history / record_step reflect the post-step values.
template <class System>
void SolverInternal<System>::step_euler(System& system, double time_max_) {
  set_time_max(time_max_);
  const double h = time_max - time;
  // Derivatives at the current state (also sets the system to y at this time).
  ode::derivs(system, y, dydt_in, time);
  const size_t size = y.size();
  for (size_t i = 0; i < size; ++i) {
    y[i] += h * dydt_in[i];
  }
  time = time_max;
  // Settle the system onto the new state at the new time.
  ode::internal::set_ode_state(system, y, time);
  push_step(system, time, h);
  dydt_in_is_clean = false;
}

// After `stepper.step()`, the GSL checks to see if the step succeeded
// (some steppers look like they fail for non-user function error),
// and the divides the step size by 2.  If it fails with `EFAULT` or
// `EBADFUNC`, then it aborts.  The only place that errors are
// actually checked in the user function, and the two errors that
// cause abort are the only two that should be thrown there.
//
// There are several different logical step sizes:
//
// 1. this->step_size_last: Size of the last successful step last
//    time, or a suggestion of one.  This will get updated as leave
//    the function only if (1) the step is successful and (2) if we're
//    not in the final step.  It's not actually quite the size of the
//    last step, either -- it's the size that the controller suggested
//    updating the step size too after the last current step.
//
// 2. step_size: The size that the current iteration actually advanced
//    the system (or will) via `stepper.step`.
//
// 3. step_size_next: The size of the proposed next step (or retry of
//    the current step).
template <class System>
void SolverInternal<System>::step(System& system) {
  const double time_orig = time, time_remaining = time_max - time;
  double step_size = step_size_last;


  // Save y in case of failure in a step (recall that stepper.step
  // changes 'y')
  const state_type y_orig = y;
  const size_t size = y.size();

  // Compute the derivatives at the beginning.
  setup_dydt_in(system);

  while (true) {
    // Does this appear to be the last step before reaching `time_max`?
    const bool final_step = step_size > time_remaining;
    if (final_step) {
      step_size = time_remaining;
    }

    // Beyond being inaccurate, a step can be *invalid* in two ways, and both are
    // rejections rather than failures: y_orig is right here, and a smaller step
    // usually lands inside the domain (#55).
    //
    //   1. A stage throws util::DomainError. This is how a model normally reports
    //      an out-of-domain state, and until now such a throw escaped this
    //      function and killed the whole solve.
    //   2. The completed step lands on a state the system's optional
    //      ode_state_valid() refuses.
    //
    // Only DomainError is caught. Anything else -- util::stop(), std::bad_alloc,
    // a logic error -- still propagates, so a bug stays a bug instead of becoming
    // "Cannot achieve the desired accuracy".
    bool invalid = false;
    std::string invalid_reason;
    try {
      stepper_step(system, time, step_size, y, yerr, dydt_in, dydt_out);
    } catch (const util::DomainError& e) {
      invalid = true;
      invalid_reason = e.what();
    }

    double step_size_next;
    if (invalid) {
      // yerr and dydt_out were never completed, so there is no error estimate to
      // form: reject on the strength of the throw alone.
      step_size_next = control.reject_step(step_size);
    } else {
      step_size_next =
        control.adjust_step_size(size, stepper_order(), step_size,
			         y, yerr, dydt_out);
      if (!state_valid(system, y)) {
        invalid = true;
        invalid_reason = "ode_state_valid() refused the state after the step";
        // Overrides whatever the error estimate concluded, including "accept".
        step_size_next = control.reject_step(step_size);
      }
    }

    if (control.step_size_shrank()) {
        // GSL checks that the step size is actually decreased.
        // Probably we can do this by comparing against hmin?  There are
        // probably loops that this will not catch, but require that
        // hmin << t
         const double time_next = time + step_size_next;
      if (step_size_next < step_size && time_next > time_orig) {
      	// Step was decreased. Undo step (resetting the state y and
        // time), and try again with the new step_size.
      	y         = y_orig;
      	time      = time_orig;
      	step_size = step_size_next;
        if (invalid) {
          // Put the system back on the restored state explicitly. After a caught
          // DomainError it is left holding whichever intermediate stage threw, and
          // if the retry goes on to raise at the minimum step size we would exit
          // with the system and y disagreeing -- the pattern behind the stale-state
          // bugs (plant#585, plant#589).
          //
          // Deliberately not done on an accuracy rejection: there the system sits
          // on the completed step's final state, the retry's stage 2 overwrites it
          // before anything reads it, and that has always been the behaviour. Doing
          // it unconditionally would add a state-set -- for plant, an environment
          // rebuild -- to every rejected step, for systems that gain nothing from
          // this feature.
          internal::set_ode_state(system, y, time);
        }
      } else {
      	// We've reached limits of machine accuracy in differences of
      	// step sizes or time (or both).
        if (invalid) {
          // Not an accuracy problem: the smallest step we are allowed to take
          // still leaves the domain. Name the reason, because it came from the
          // system and is the only description of what is actually wrong.
          util::stop("Cannot leave an invalid state at t = " +
                     util::format_double(time_orig) + ": " + invalid_reason +
                     " (step size " + util::format_double(step_size) +
                     " is already at the minimum)");
        }
        util::stop("Cannot achieve the desired accuracy");
      }
    } else {
      // We have successfully taken a step and will return.  Update
      // time to reflect this, ensuring that if we're on the last step
      // we will end up exactly at time_max.
      //
      // Suggest step size for next time-step. Change of step size is not
      //  suggested in the final step, because that step can be very
      //  small compared to previous step, to reach time_max.
      if (final_step) {
	      time = time_max;
      } else {
	      time += step_size;
	      step_size_last = step_size_next;
      }
      save_dydt_out_as_in();
      push_step(system, time, step_size);
      return; // This exits the infinite loop.
    }
  }
}

// This takes a step up to time "time_max_", regardless of what the
// integration error says.  This is used by advance_fixed
template <class System>
void SolverInternal<System>::step_to(System& system, double time_max_) {
  set_time_max(time_max_);
  const double step_size = time_max - time;
  setup_dydt_in(system);
  stepper_step(system, time, step_size, y, yerr, dydt_in, dydt_out);
  save_dydt_out_as_in();

  time = time_max;
  push_step(system, time, step_size);
}

// This takes a step of the given size, regardless of what the integration error
// says.  This is used by advance_fixed_steps.
template <class System>
void SolverInternal<System>::step_by(System& system, double step_size) {
  if (!util::is_finite(step_size)) {
    util::stop("step_size must be finite!");
  }
  if (step_size < 0.0) {
    util::stop("step_size must be greater than (or equal to) zero");
  }
  setup_dydt_in(system);
  stepper_step(system, time, step_size, y, yerr, dydt_in, dydt_out);
  save_dydt_out_as_in();

  time += step_size;
  time_max = time;
  push_step(system, time, step_size);
}

template <class System>
void SolverInternal<System>::resize(size_t size_) {
  y.resize(size_);
  yerr.resize(size_);
  dydt_in.resize(size_);
  dydt_out.resize(size_);
  stepper.resize(size_);
  rodas_stepper.resize(size_);
}

template <class System>
void SolverInternal<System>::setup_dydt_in(System& system) {
  if (stepper_can_use_dydt_in() && !dydt_in_is_clean) {
    // The full derivs() -- set the state, then read the rates -- rather than
    // reading the rates alone. y is the solver's own state and need not be the
    // one the system currently holds, so the state has to be re-established
    // before the rates mean anything.
    ode::derivs(system, y, dydt_in, time);
    dydt_in_is_clean = true;
  }
}

template <class System>
void SolverInternal<System>::save_dydt_out_as_in() {
  if (stepper_first_same_as_last()) {
    dydt_in = dydt_out;
    dydt_in_is_clean = true;
  } else {
    dydt_in_is_clean = false;
  }
}

template <typename System>
void SolverInternal<System>::set_time(double t) {
  const int ulp = 2; // units in the last place (accuracy)
  if (prev_steps.size() > 0 &&
      !util::almost_equal(prev_steps.back().time, t, ulp))
  {
    util::stop("Time does not match previous (delta = " +
               util::format_double(prev_steps.back().time - t) +
               "). Reset solver first.");
  }
  time = t;
  if (prev_steps.empty()) { // only if first time (avoids duplicate times)
    // No step reached the initial time, so it records no size. The state is
    // recorded by set_state_from_system, which calls this and then holds it.
    prev_steps.push_back(
      step_record{time, std::numeric_limits<double>::quiet_NaN(), state_type()});
  }
}

template <class System>
void SolverInternal<System>::set_time_max(double time_max_) {
  if (!util::is_finite(time_max_)) {
    util::stop("time_max must be finite!");
  }
  if (time_max_ < time) {
    util::stop("time_max must be greater than (or equal to) current time");
  }
  time_max = time_max_;
}

}
}

#endif