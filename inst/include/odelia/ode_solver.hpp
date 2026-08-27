#ifndef ODELIA_ODE_SOLVER_HPP_
#define ODELIA_ODE_SOLVER_HPP_

#include <odelia/ode_solver_internal.hpp>
#include <odelia/adjoint.hpp>
#include <algorithm>
#include <utility>
#include <XAD/XAD.hpp>
#include <cmath>
#include <span>
#include <vector>

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

  // Record the wider state an insertion just reached, on the row it followed.
  void push_inserted() { solver.push_inserted(system); }

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

  // Step over a schedule, landing on each of its times: at the recorded size
  // where one is known, and to the time itself where it is not. Not by
  // differencing the times, because a size differenced back out of two recorded
  // times is not the size that was taken; and not by adding sizes, which arrives
  // a rounding short of where the run landed.
  void advance_recorded(const std::vector<recorded_step>& steps)
  {
    if (steps.empty())
    {
      util::stop("'steps' must be a recording of at least length 1");
    }
    if (!std::isnan(steps.front().step_size))
    {
      util::stop("The first recorded step must have a NaN size, being the start "
                 "that no step reached");
    }

    if (collect)
    {
      history.push_back(system);
    }

    for (std::size_t k = 1; k < steps.size(); ++k)
    {
      if (std::isnan(steps[k].step_size))
      {
        solver.step_to(system, steps[k].time);
      }
      else
      {
        solver.step_by(system, steps[k].step_size, steps[k].time);
      }
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
  // The record the run kept: one row per accepted step, each carrying its time,
  // the size that reached it and the state there. What a sweep reads, and the
  // only place it reads them from.
  std::span<const ode::step_record<System>> recording() const {
    return solver.recording();
  }
  std::size_t recorded_steps() const { return solver.recorded_steps(); }
  // The schedule a replay of this run would take. Read off the record, so the
  // time and the size that reached it cannot be paired across two runs.
  std::vector<ode::recorded_step> schedule() const { return solver.schedule(); }

  // Carry lambda back over recorded rows k_last down to k_first + 1, highest
  // first, adding each row's parameter contribution into `parameter_adjoint`.
  //
  // Wherever the state changed width inside that range the descent stops, narrows
  // the System across it and transposes the map that widened it, so the rows it
  // carries arrive by the same route a step's do. Where nothing widened this is one
  // lift and one descent -- which is what a System of fixed width gets, and what
  // every range this splits into gets.
  //
  // `extra_splits` names further rows to stop at. The adjoint carried across such a
  // stop is the same one either way, so a caller can ask for a split and compare: a
  // check, not a choice.
  //
  // The System is put where each range needs it and left where the run left it,
  // so a sub-range can be swept without the caller positioning anything and the
  // whole can be swept again afterwards.
  //
  // Returns how many ranges carried a step.
  std::size_t solve_adjoint(ode::adjoint_rows& lambda,
                            ode::adjoint_rows& parameter_adjoint,
                            size_t k_first, size_t k_last,
                            const std::vector<size_t>& extra_splits = {})
  {
    using scalar = ode::active_scalar<double>;
    const std::span<const ode::step_record<System>> rec = recording();
    if (rec.size() < 2) {
      util::stop("solve_adjoint: no recorded steps to sweep; run the adaptive "
                 "pass first");
    }
    if (k_first >= k_last || k_last >= rec.size()) {
      util::stop("the adjoint segment is not a range of recorded steps");
    }

    // Where the descent stops. An insertion AT k_first is a stop: it carries the
    // state the run began from into the first width, and nothing below it is swept.
    // A split there would cut nothing, so it is not.
    std::vector<size_t> stops;
    for (const size_t at : ode::insertion_rows(rec)) {
      if (at >= k_first && at < k_last) {
        stops.push_back(at);
      }
    }
    for (const size_t at : extra_splits) {
      if (at > k_first && at < k_last) {
        stops.push_back(at);
      }
    }
    // Sorted and deduplicated, so a split landing on a widening is that widening
    // rather than a second stop at the same row.
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());

    // ⚠️ THE WIDTH ON EXIT IS A PROMISE, AND A THROW IS AN EXIT. The descent starts
    // at the run's own width and narrows as it goes, so a sweep abandoned high up
    // leaves the System at its widest -- where every caller's tail widens back from
    // the lowest and reads the mismatch as a length error one call later, naming
    // neither this walk nor what refused.
    struct restore_on_exit {
      System& sys;
      std::span<const ode::step_record<System>> rec;
      ~restore_on_exit() {
        // This runs with another exception possibly in flight, so a failure here
        // cannot be raised: it would end the process rather than the call that is
        // already failing.
        try {
          ode::be_at_step(sys, rec, rec.size() - 1);
        } catch (...) {
        }
      }
    } restore{system, rec};

    // One tape for the whole descent, held active across every recording it takes.
    // Clearing between recordings keeps the capacity the largest of them grew,
    // where one tape per recording regrows it every time.
    ode::adjoint_tape<double> tape(false);
    ode::tape_scope<ode::adjoint_tape<double>> running{tape};

    // One range per width, highest first: the System is put at the range's top, so
    // whoever narrows it is whoever lifts on it and no descent inherits a width
    // another call left behind.
    size_t swept = 0;
    for (size_t j = stops.size() + 1; j-- > 0;) {
      const size_t hi = (j == stops.size()) ? k_last : stops[j];
      const size_t lo = (j == 0) ? k_first : stops[j - 1];
      ode::be_at_step(system, rec, hi);

      // The insertion that widened `hi`, transposed at the width below it -- which
      // is the width the range below runs at. The topmost range ends at k_last,
      // which no insertion inside this range widened.
      //
      // ⚠️ IT LIFTS ITS OWN, AND THAT IS NOT A SHARING SOMEONE MISSED. The map is
      // `inserted_state`, which sets the narrow state, pushes the nodes and reports
      // the wider state -- so transposing it leaves the System it ran on WIDER than
      // the range below is about to be swept at. One lift handed to both reads past
      // the end of every state the descent then loads: unmapped memory where the
      // widths differ most, and finite nonsense where they do not. Sharing it needs
      // `inserted_state` to stop mutating what it reports.
      if (j < stops.size() && !rec[hi].inserted.empty()) {
        const double when = rec[hi].time;
        auto insert = [&](auto& active_system,
                          typename std::vector<scalar>::const_iterator x,
                          std::vector<scalar>& y) -> void {
          ode::inserted_state(active_system, when, x, y);
        };
        ode::lifted_system<System> widened{system, tape};
        ode::adjoint_rows narrowed;
        ode::state_and_parameter_adjoints(widened, rec[hi].state, lambda, insert,
                                         narrowed, parameter_adjoint);
        lambda = std::move(narrowed);
      }

      // A range with no step in it cuts nothing, which is what an insertion at the
      // range's first row gives.
      if (lo < hi) {
        ode::lifted_system<System> active{system, tape};
        sweep_range(active, rec, lambda, parameter_adjoint, lo, hi);
        ++swept;
      }
    }
    return swept;
  }

  // The whole recording.
  std::size_t solve_adjoint(ode::adjoint_rows& lambda,
                            ode::adjoint_rows& parameter_adjoint,
                            const std::vector<size_t>& extra_splits = {})
  {
    return solve_adjoint(lambda, parameter_adjoint, 0, recording().size() - 1,
                         extra_splits);
  }

  // Rate evaluations the sweeps since the last clear have recorded: six a step,
  // whatever the seed count. A term entering once a step where it belongs once a
  // stage divides this by six, and no gradient check can see that, because a
  // tangent and a sweep apply the same multiplier.
  std::size_t recorded_rates() const { return solver.recorded_rates(); }
  void clear_recorded_rates() { solver.clear_recorded_rates(); }

  // Should we record history at every step?
  // TODO: should this be part of ode_solver?
std::vector<System> history;

private:
  // One range, at the width the lift handed in was taken at: the caller narrowed
  // the System and lifted it there, so one copy serves every step in this range and
  // each recording releases its slots before taking the next.
  //
  // Every state visited has to be that width, so a range a widening crossed is
  // refused by the length check inside the loop rather than swept at one width.
  void sweep_range(ode::lifted_system<System>& active,
                   std::span<const ode::step_record<System>> rec,
                   ode::adjoint_rows& lambda, ode::adjoint_rows& parameter_adjoint,
                   size_t k_first, size_t k_last)
  {
    if (lambda.empty()) {
      util::stop("solve_adjoint: needs at least one seed");
    }
    // Named, because a bare length mismatch here is read as the caller's and says
    // nothing about the seam it is really about: a batch carried at one width
    // against a System left at another. One width for every row, so this is asked
    // of the batch and not of each seed in it.
    if (lambda.width() != system.ode_size()) {
      util::stop("solve_adjoint: the seeds are " +
                 util::to_string(static_cast<int>(lambda.width())) +
                 " wide against a System of " +
                 util::to_string(static_cast<int>(system.ode_size())) +
                 ", so the two are not at the same widening");
    }
    ode::adjoint_rows lambda_in;
    for (size_t k = k_last; k > k_first; --k) {
      // What the run's step k ran from, which is the wider state where an
      // insertion followed row k - 1 and that row's own where none did.
      const state_type<System>& from = rec[k - 1].ran_from();
      util::check_length(from.size(), system.ode_size());
      solver.step_adjoint(active, k, rec[k - 1].time, rec[k].step_size, from,
                          lambda, lambda_in, parameter_adjoint);
      // Swapped rather than moved from: a move leaves the buffer this step wrote
      // into empty, so the next step allocates one the same size again. Swapping
      // hands it the row above's, which the sweep refills rather than regrows.
      std::swap(lambda, lambda_in);
    }
  }

  bool collect;
  System system;
  OdeControl control_;
  SolverInternal<System> solver;
};
}
}
#endif
