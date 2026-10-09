#ifndef ODELIA_ODE_SOLVER_HPP_
#define ODELIA_ODE_SOLVER_HPP_

#include <odelia/ode_solver_internal.hpp>
#include <odelia/adjoint.hpp>
#include <algorithm>
#include <utility>
#include <XAD/XAD.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

  // Synchronize internal ODE buffers from the current system state without
  // resetting solver history/step-size state. `seed` is what the evaluation there
  // starts from, where a recording is walked.
  void set_state_from_system(const solved_values_t<System>* seed = nullptr)
  {
    solver.set_state_from_system(system, seed);
  }

  // Record the wider state an insertion just reached, on the row it followed.
  void push_insertion() { solver.push_insertion(system); }

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

  // Walk rows: a program's, or a recording's, of this System or of the same System
  // at another scalar. A step row is taken at the size the row records, and to its
  // time where it records none; an insertion row applies `insert` to the state
  // held and records the insertion as the run it follows did.
  //
  // The step sizes are replayed rather than the times: a size differenced back out
  // of two recorded times is not the size that was taken, since fl(fl(t + h) - t)
  // is not h, and a walk that chose its own would be differentiating a controller
  // the model does not contain.
  //
  // A recording's rows travel WITH the steps: each evaluation's own slot starts as
  // a copy of the row's, so a System reads what the recorded run left there and
  // stores what it solves for itself.
  //
  // ⚠️ `rows[k].solved` is what the step that REACHED `rows[k].time` solved, which
  // is the pairing `solve_adjoint` walks. Off by one here and every evaluation
  // reads its neighbour's row, finitely.
  template <class Rows, class Insert>
  void advance_recorded(const Rows& rows, Insert&& insert)
  {
    if (rows.size() == 0)
    {
      util::stop("'rows' must hold at least the row the walk starts from");
    }
    // The first row is where the caller put the System: a run's start, or an
    // insertion already applied. A step there would be skipped.
    if (!rows[0].insertion && !std::isnan(rows[0].step_size))
    {
      util::stop("A walk's first row is where it starts, so it must be a run's "
                 "start or an insertion, not a step of size " +
                 util::format_double(rows[0].step_size));
    }
    // The start's own evaluation, seeded where the rows carry one.
    if constexpr (requires { rows[0].solved; })
    {
      set_state_from_system(&rows[0].solved.at_state);
    }

    if (collect)
    {
      history.push_back(system);
    }

    for (std::size_t k = 1; k < rows.size(); ++k)
    {
      const auto& row = rows[k];
      if (row.insertion)
      {
        insert(system, row.time);
        push_insertion();
        if constexpr (requires { row.solved; })
        {
          set_state_from_system(&row.solved.at_state);
        }
        else
        {
          set_state_from_system();
        }
        continue;
      }
      if constexpr (requires { row.solved; })
      {
        // A recorded step was taken, so it has a size; stepping to its time
        // instead would leave the row unread.
        if (std::isnan(row.step_size))
        {
          util::stop("A recorded step carries the size it took; row " +
                     util::to_string(k) + " has none");
        }
        solver.step_by(system, row, &row);
      }
      else if (std::isnan(row.step_size))
      {
        solver.step_to(system, row.time);
      }
      else
      {
        solver.step_by(system, row);
      }
      if (collect)
      {
        history.push_back(system);
      }
    }
  }

  // The same walk, with each insertion the System's own.
  template <class Rows>
  void advance_recorded(const Rows& rows)
  {
    advance_recorded(rows, [](System& sys, double time) {
      ode::apply_insertion(sys, time);
    });
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
  // The schedule a replay of this run would take. Read off the record, so the
  // time and the size that reached it cannot be paired across two runs.
  std::vector<ode::instruction> schedule() const { return solver.schedule(); }

  // Carry lambda back over recorded rows k_last down to k_first + 1, highest
  // first, adding each row's parameter contribution into `parameter_adjoint`.
  //
  // Wherever the state changed width inside that range the descent stops, narrows
  // the System across it and transposes the map that widened it, so the rows it
  // carries arrive by the same route a step's do. Where nothing widened this is one
  // rebind and one descent -- which is what a System of fixed width gets, and what
  // every range this splits into gets.
  //
  // `extra_stops` names further rows to stop at. The adjoint carried across such a
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
                            const std::vector<size_t>& extra_stops = {})
  {
    using scalar = ode::active_scalar<double>;
    const std::span<const ode::step_record<System>> rec = recording();
    if (rec.size() < 2) {
      util::stop("solve_adjoint: no recorded steps to sweep; run the adaptive "
                 "pass first");
    }
    if (k_first >= k_last || k_last >= rec.size()) {
      util::stop("the adjoint range is not a range of recorded steps");
    }

    // A insertion is a row the sweep carries an adjoint ACROSS, so it cannot be one
    // the descent starts at or the one it is left standing on. Neither happens on a
    // recording a run made -- the schedule puts every introduction at the start of
    // an interval that then steps -- and both are checked here rather than at each
    // be_at_step, because the width this leaves the System at is a promise and the
    // call that restores it cannot raise.
    if (rec.back().insertion || rec[k_last].insertion) {
      util::stop("solve_adjoint: a recording cannot end at an insertion, because "
                 "nothing stepped away from the state it made");
    }

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
    // whoever narrows it is whoever rebinds on it and no descent inherits a width
    // another call left behind.
    //
    // A stop is either an insertion the run recorded or a cut a caller asked for, and
    // they leave the descent in different places. A cut is a row the sweep resumes
    // at; an insertion is a row the sweep carries the adjoint ACROSS, so it resumes
    // one row below, on the state the map ran on.
    size_t swept = 0;
    size_t hi = k_last;
    const auto sweep_down_to = [&](size_t lo) -> void {
      ode::be_at_step(system, rec, hi);
      // A range with no step in it cuts nothing, which is what two stops in a row
      // gives.
      if (lo < hi) {
        sweep_range(tape, rec, lambda, parameter_adjoint, lo, hi);
        ++swept;
      }
    };

    // The rows this can stop at are the rows strictly inside the range, which is
    // what the bounds say, and each is reached once. A row that is both an insertion
    // and a cut is the insertion, which is what keeps a cut free of a map.
    for (size_t at = k_last; at-- > k_first + 1;) {
      const bool insertion = rec[at].insertion;
      if (!insertion && std::find(extra_stops.begin(), extra_stops.end(), at) ==
                           extra_stops.end()) {
        continue;
      }
      sweep_down_to(at);
      if (!insertion) {
        hi = at;
        continue;
      }
      // The map that widened `at`, transposed at the width below it -- which is
      // the width the range below runs at, and the width the row below holds. The
      // run applied it after the evaluation the row below recorded, so that is
      // repeated first.
      //
      // Its active System is its own and dies with it, because applying the map is
      // what widens the System: what this records on cannot be swept at the width
      // it started from.
      ode::be_at_step(system, rec, at - 1);
      const double when = rec[at].time;
      const double below_time = ode::at_state_time(rec, at - 1);
      auto insert = [&](auto& sys,
                        typename std::vector<scalar>::const_iterator x,
                        std::vector<scalar>& y) -> void {
        const std::vector<scalar> below(
            x, x + static_cast<std::ptrdiff_t>(sys.ode_size()));
        std::vector<scalar> rates(below.size());
        ode::derivs(sys, below, rates, below_time,
                    std::as_const(rec[at - 1].solved.at_state));
        ode::apply_insertion(sys, when);
        y.assign(sys.ode_size(), scalar(0.0));
        sys.ode_state(y.begin());
      };
      ode::active_system<System> widened{system, tape};
      ode::adjoint_rows narrowed;
      ode::state_and_parameter_adjoints(widened, rec[at - 1].state, lambda, insert,
                                       narrowed, parameter_adjoint);
      lambda = std::move(narrowed);
      hi = at - 1;
    }
    sweep_down_to(k_first);
    return swept;
  }

  // The whole recording.
  std::size_t solve_adjoint(ode::adjoint_rows& lambda,
                            ode::adjoint_rows& parameter_adjoint,
                            const std::vector<size_t>& extra_stops = {})
  {
    return solve_adjoint(lambda, parameter_adjoint, 0, recording().size() - 1,
                         extra_stops);
  }

  // Rate evaluations the sweeps since the last clear have recorded; Step::
  // recorded_rates says what the number means and what it catches.
  std::size_t recorded_rates() const { return solver.recorded_rates(); }
  void clear_recorded_rates() { solver.clear_recorded_rates(); }

  // How the attempts at each error-controlled step ended; step_outcomes says
  // what each count covers.
  const ode::step_outcomes& outcomes() const { return solver.outcomes(); }
  // By block, the steps this run split at a sign change since the last reset.
  const std::vector<std::size_t>& splits_by_block() const {
    return solver.splits_by_block();
  }

  // Should we record history at every step?
  // TODO: should this be part of ode_solver?
std::vector<System> history;

private:
  // One range, at the width the caller left the System on: it is rebound here, so
  // one copy serves every step in this range, each recording releases its slots
  // before taking the next, and no active System arrives from a call that widened
  // it.
  //
  // Every state visited has to be that width, so a range an insertion crossed is
  // refused by the length check inside the loop rather than swept at one width.
  void sweep_range(ode::adjoint_tape<double>& tape,
                   std::span<const ode::step_record<System>> rec,
                   ode::adjoint_rows& lambda, ode::adjoint_rows& parameter_adjoint,
                   size_t k_first, size_t k_last)
  {
    if (lambda.empty()) {
      util::stop("solve_adjoint: needs at least one seed");
    }
    ode::active_system<System> active{system, tape};
    // Asked of the System the recordings are taken on, which is the width every
    // state below has to be loaded at. Named, because a bare length mismatch here
    // is read as the caller's and says nothing about the seam it is really about:
    // a batch carried at one width against a System at another. One width for
    // every row, so this is asked of the batch and not of each seed in it.
    if (lambda.width() != active.system.ode_size()) {
      util::stop("solve_adjoint: the seeds are " +
                 util::to_string(static_cast<int>(lambda.width())) +
                 " wide against a System of " +
                 util::to_string(static_cast<int>(active.system.ode_size())) +
                 ", so the two are not at the same insertion");
    }
    // ⚠️ THE DESCENT STOPS AT THE FIRST NON-FINITE ENTRY, and must not be changed
    // to carry it. This hands its caller one number per input, so an overflow the
    // descent picked up three thousand steps ago would arrive indistinguishable
    // from a NaN the last step made -- and a caller polling for a declared
    // refusal would find none, which is the one failure this gradient must never
    // produce. Raised as AdjointRangeError, which the consumer turns into a
    // refusal of every metric.
    //
    // ODELIA_ADJOINT_TRACE=steps prints the magnitude at every step, which is
    // what says whether the descent compounded into the failure or met it. One
    // line per recorded step is thousands of them, so it is not the default; the
    // magnitude one step above is in the message either way.
    static const char* const trace = std::getenv("ODELIA_ADJOINT_TRACE");
    double worst_before = 0.0;
    const bool per_step = trace != nullptr && std::strcmp(trace, "steps") == 0;
    ode::adjoint_rows lambda_in;
    for (size_t k = k_last; k > k_first; --k) {
      // What the run's step k ran from: the row below it, whether that row is a
      // step's landing or an insertion's output.
      const state_type<System>& from = rec[k - 1].state;
      util::check_length(from.size(), active.system.ode_size());
      solver.step_adjoint(active, rec[k - 1].solved.at_state,
                          ode::at_state_time(rec, k - 1), rec[k].solved,
                          rec[k - 1].time, rec[k].step_size, from,
                          lambda, lambda_in, parameter_adjoint, rec[k].subsystem);
      // Swapped rather than moved from: a move leaves the buffer this step wrote
      // into empty, so the next step allocates one the same size again. Swapping
      // hands it the row above's, which the sweep refills rather than regrows.
      std::swap(lambda, lambda_in);
      double worst_here = 0.0;
      for (size_t m = 0; m < lambda.rows(); ++m) {
        const std::span<double> row = lambda[m];
        for (size_t j = 0; j < row.size(); ++j) {
          if (!std::isfinite(row[j])) {
            util::stop_adjoint_range(
                "the adjoint left the representable range at step " +
                util::to_string(k) + " of [" + util::to_string(k_first) + ", " +
                util::to_string(k_last) + "], t=" +
                util::format_double(rec[k - 1].time) + ", h=" +
                util::format_double(rec[k].step_size) + ": seed " +
                util::to_string(m) + "'s entry " + util::to_string(j) + " of " +
                util::to_string(row.size()) + " is " +
                util::format_double(row[j]) + ", where the step above carried " +
                util::format_double(worst_before) +
                ". A descent is a product of step Jacobians and has no error "
                "control, so it can pass outside the range of the answer it "
                "returns; this one did not come back.");
          }
          const double at = std::fabs(row[j]);
          if (at > worst_here) worst_here = at;
        }
      }
      if (per_step) {
        std::fprintf(stderr,
                     "ODELIA_ADJOINT_TRACE step %zu of [%zu, %zu] t=%.12g "
                     "h=%.6g worst |lambda| %.6g\n",
                     k, k_first, k_last, rec[k - 1].time, rec[k].step_size,
                     worst_here);
      }
      worst_before = worst_here;
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
