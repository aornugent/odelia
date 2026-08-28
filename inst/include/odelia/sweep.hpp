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

// Carrying an adjoint back over a recording whose state width changed. One tape,
// held active for the whole descent; one System rebound per width, releasing its
// slots before each recording clears the tape.

namespace odelia {
namespace ode {

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
    apply_insertion(system, rec[start].time, rec[start].state.begin(), base);
    util::check_length(base.size(), rec[start].inserted.size());
    return rec[start].time;
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
//
// The head entry carries the solver's own time rather than the row's, which is
// what lets a caller replay a perturbed state at that time. Everything else is
// the recording, junctions included -- the solver executes those, so this builds
// a program and makes one call.
template <class Solver, class Record>
void advance_over_insertions(
    Solver& forward, std::span<const Record> rec, std::size_t from_segment,
    std::size_t first) {
    const std::vector<std::size_t> rows = insertion_rows(rec);
    if (from_segment > rows.size()) {
        util::stop("advance_over_insertions: the recording has no such segment");
    }
    // The junction at `first` is this walk's only where `first` is the segment it
    // starts at; a caller resuming mid-recording has had it applied by whatever
    // put the System there.
    const bool pending =
        from_segment < rows.size() && rows[from_segment] == first;
    std::vector<recorded_step> program;
    program.reserve(rec.size() - first);
    program.push_back(
        {forward.time(), std::numeric_limits<double>::quiet_NaN(), pending});
    for (std::size_t k = first + 1; k < rec.size(); ++k) {
        program.push_back(rec[k]);
    }
    forward.advance_recorded(program);
}

}  // namespace ode
}  // namespace odelia

#endif
