// -*-c++-*-
#ifndef ODELIA_SWEEP_HPP_
#define ODELIA_SWEEP_HPP_

#include <odelia/adjoint.hpp>
#include <odelia/ode_solver.hpp>
#include <algorithm>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// Carrying an adjoint back over a recording whose state width changed. The
// System is lifted per recording and the tape cleared between them, which is the
// one tape discipline in this library.

namespace odelia {
namespace ode {

// The rows an insertion followed: the rows carrying the wider state the run
// reached before the step above them. Read off the record, because the forward
// pass knew it was inserting and wrote it there.
//
// A walk that recovered these by scanning for a width which grew had to refuse a
// width which shrinks, because an inference can be wrong where a recorded fact
// cannot.
template <class Record>
std::vector<std::size_t> insertion_rows(std::span<const Record> rec) {
    if (rec.size() < 2) {
        util::stop("insertion_rows: no recorded steps to sweep");
    }
    std::vector<std::size_t> ret;
    for (std::size_t k = 0; k + 1 < rec.size(); ++k) {
        if (!rec[k].inserted.empty()) {
            ret.push_back(k);
        }
    }
    return ret;
}

// Put the System on the state the run recorded at `step`. The System reconciles
// itself to that time -- which insertions had happened by then is derived from the
// schedule it was driven by, not handed to it -- and the width it arrives at is
// checked against the width recorded there.
//
// Idempotent, and that is the whole reason it is a load: the System reconciles to
// the step rather than stepping toward it, so arriving twice is arriving once and
// a walk can be run again over the recording it has already walked.
template <class System>
void be_at_step(System& system, std::span<const step_record<System>> rec,
                std::size_t step) {
    if (step >= rec.size()) {
        util::stop("be_at_step: step " +
                   util::to_string(static_cast<int>(step)) +
                   " is outside a recording of " +
                   util::to_string(static_cast<int>(rec.size())) + " steps");
    }
    system.set_recorded_state(rec[step].state, rec[step].time);
    // Named, because a bare length mismatch reads as a caller's error one call
    // away and says nothing about which walk or which step refused.
    if (system.ode_size() != rec[step].state.size()) {
        util::stop("be_at_step: reconciled to " +
                   util::to_string(static_cast<int>(system.ode_size())) +
                   " wide at step " +
                   util::to_string(static_cast<int>(step)) + " against " +
                   util::to_string(static_cast<int>(rec[step].state.size())) +
                   " recorded there");
    }
}

// The state a segment's first step ran from, and the time it ran at. It is what
// the run reached between a widening and the step after it, so no record holds
// it and it is replayed. The System is left at that segment's width, because a
// caller reading this state wants to run from it.
template <class System>
double state_at_segment(System& system,
                        std::span<const step_record<System>> rec,
                        std::size_t segment,
                        std::vector<double>& base, std::size_t& start) {
    const std::vector<std::size_t> rows = insertion_rows(rec);
    if (segment > rows.size()) {
        util::stop("state_at_segment: the recording has no such segment");
    }
    if (segment == 0) {
        be_at_step(system, rec, 0);
        start = 0;
        base.assign(system.ode_size(), 0.0);
        system.ode_state(base.begin());
        return rec[0].time;
    }
    // The insertion's own map at the state below it, which is the same map the
    // sweep transposes and the same one the run recorded the result of -- so the
    // width is checked against what the record holds.
    start = rows[segment - 1];
    be_at_step(system, rec, start);
    system.inserted_state(rec[start].time, rec[start].state.begin(), base);
    util::check_length(base.size(), rec[start].inserted.size());
    return rec[start].time;
}

// Carry lambda back over a recording, one row at a time, highest first. Where an
// insertion widened a row, the System is narrowed across it and the map that
// widened it is transposed there, so the rows it carries reach
// `parameter_adjoint` by the same route a step's do.
//
// `extra_splits` names rows at which the descent stops and resumes. The adjoint
// recursion is linear in the step, so a split must change no number; it is a
// check, not a choice. Returns how many ranges were swept, which is one per
// stretch between stops and is therefore the split count plus the insertion
// count plus one, less any stretch with no step in it.
template <class Solver>
std::size_t solve_adjoint_over_insertions(
    Solver& solver,
    std::span<const step_record<typename Solver::system_type>> rec,
    row_batch& lambda, row_batch& parameter_adjoint,
    const std::vector<std::size_t>& extra_splits = {}) {
    using scalar = active_scalar<double>;
    using record = step_record<typename Solver::system_type>;
    auto& system = solver.get_system_ref();

    // Where the descent stops: every row an insertion widened, and every row the
    // caller asked to stop at. One sorted list, so a split that lands on a
    // widening is that widening rather than a second stop at the same row.
    std::vector<std::size_t> stops = insertion_rows(rec);
    for (const std::size_t at : extra_splits) {
        if (at > 0 && at + 1 < rec.size()) {
            stops.push_back(at);
        }
    }
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());

    // ⚠️ THE WIDTH ON EXIT IS A PROMISE, AND A THROW IS AN EXIT. The descent
    // below starts at the run's own width and narrows as it goes, so a sweep
    // abandoned high up leaves the System at its widest -- where every caller's
    // tail widens back from the lowest and reads the mismatch as a length error
    // one call later, naming neither this walk nor what refused.
    struct restore_on_exit {
        // Named apart from what it binds to: a member named `system` would change
        // what `system` means inside its own declaration.
        decltype(system) sys;
        std::span<const record> rec;
        ~restore_on_exit() {
            // This runs with another exception possibly in flight, so a failure
            // here cannot be raised: it would end the process rather than the
            // call that is already failing.
            try {
                be_at_step(sys, rec, rec.size() - 1);
            } catch (...) {
            }
        }
    } restore{system, rec};

    std::size_t swept = 0;
    std::size_t upper = rec.size() - 1;
    // One tape for the whole walk, held active across every recording it takes.
    // Clearing between recordings keeps the capacity the largest of them grew,
    // where one tape per recording regrows it every time.
    typename scalar::tape_type tape(false);
    tape_scope<typename scalar::tape_type> running{tape};
    for (std::size_t j = stops.size(); j-- > 0;) {
        const std::size_t at = stops[j];
        // A stop with no step above it cuts nothing, which is what an insertion
        // at the last recorded row gives.
        if (at < upper) {
            solver.solve_adjoint(tape, rec, lambda, parameter_adjoint, at, upper);
            ++swept;
        }
        upper = at;
        if (rec[at].inserted.empty()) {
            continue;  // the caller's own split; the width does not change here
        }
        be_at_step(system, rec, at);
        const double when = rec[at].time;
        auto insert = [&](auto& active_system,
                          typename std::vector<scalar>::const_iterator x,
                          std::vector<scalar>& y) -> void {
            active_system.inserted_state(when, x, y);
        };
        // Lifted at the width below the widening, which is where be_at_step has
        // just put the System, and used for this one recording.
        lifted_system<typename Solver::system_type> active{system, tape};
        row_batch narrowed;
        state_and_parameter_adjoints(active, rec[at].state, lambda, insert,
                                     narrowed, parameter_adjoint);
        lambda = std::move(narrowed);
    }
    // An insertion at the first recorded row leaves nothing below it, which is
    // what a run from an empty state gives.
    if (upper > 0) {
        solver.solve_adjoint(tape, rec, lambda, parameter_adjoint, 0, upper);
        ++swept;
    }
    return swept;
}

// Step `forward` over the recording from `from_segment` on, at the sizes the run
// took, widening where the run widened. `first` is the recorded row the walk
// begins at, which the caller has already put the System on. The schedule is read
// off the recording, because a recording row is a schedule row plus its state.
//
// The step sizes are replayed rather than the times: a size differenced back out
// of two recorded times is not the size that was taken, since fl(fl(t + h) - t)
// is not h, and a walk that chose its own would be differentiating a controller
// the model does not contain.
template <class Solver, class Record>
void advance_over_insertions(
    Solver& forward, std::span<const Record> rec, std::size_t from_segment,
    std::size_t first) {
    const std::vector<std::size_t> rows = insertion_rows(rec);
    if (from_segment > rows.size()) {
        util::stop("advance_over_insertions: the recording has no such segment");
    }
    for (std::size_t j = from_segment; j <= rows.size(); ++j) {
        const std::size_t last = j < rows.size() ? rows[j] : rec.size() - 1;
        // The first entry is the start no step reached, which is how a recorded
        // run reads back. A recording row is a schedule row plus its state, so
        // the schedule is read off it rather than built beside it.
        std::vector<recorded_step> segment_steps{
            {forward.time(), std::numeric_limits<double>::quiet_NaN()}};
        for (std::size_t k = first + 1; k <= last; ++k) {
            segment_steps.push_back(rec[k]);
        }
        if (segment_steps.size() > 1) {
            forward.advance_recorded(segment_steps);
        }
        if (j == rows.size()) {
            break;
        }
        // The same map the sweep transposes, so a tangent traverses exactly the
        // function under test rather than a second spelling of it. It leaves the
        // System holding what it added, which is what this wants; the wider state
        // it also reports is read from the System below.
        auto& sys = forward.get_system_ref();
        using value_type = typename Solver::value_type;
        std::vector<value_type> before(sys.ode_size());
        sys.ode_state(before.begin());
        std::vector<value_type> widened;
        sys.inserted_state(rec[last].time, before.begin(), widened);
        forward.set_state_from_system();
        first = last;
    }
}
}
}

#endif
