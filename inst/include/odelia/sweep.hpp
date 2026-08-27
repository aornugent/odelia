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
// held active for the whole descent; one System lifted per width, releasing its
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
template <class Solver, class Record>
void advance_over_insertions(
    Solver& forward, std::span<const Record> rec, std::size_t from_segment,
    std::size_t first) {
    const std::vector<std::size_t> rows = insertion_rows(rec);
    if (from_segment > rows.size()) {
        util::stop("advance_over_insertions: the recording has no such segment");
    }
    // Held across the walk: the state below each insertion, and the wider one the
    // map reports and this does not read -- the System is left holding it, which
    // is what this wants.
    using value_type = typename Solver::value_type;
    std::vector<value_type> before;
    std::vector<value_type> widened;
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
        // function under test rather than a second spelling of it.
        auto& sys = forward.get_system_ref();
        before.assign(sys.ode_size(), value_type(0.0));
        sys.ode_state(before.begin());
        apply_insertion(sys, rec[last].time, before.begin(), widened);
        forward.set_state_from_system();
        first = last;
    }
}
}
}

#endif
