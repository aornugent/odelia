// -*-c++-*-
#ifndef ODELIA_SWEEP_HPP_
#define ODELIA_SWEEP_HPP_

#include <odelia/adjoint.hpp>
#include <odelia/ode_solver.hpp>
#include <span>
#include <utility>
#include <vector>

// Carrying an adjoint back over a recording whose state width changed. The tape
// discipline here is adjoint.hpp's: the System is lifted per recording, so the
// tape is cleared between them.
//
// Apart from calibration.hpp, which records on a System it keeps and therefore
// cannot clear. Holding the two in one header meant a reader had to track which
// regime a function belonged to.

namespace odelia {
namespace ode {

// A range of the recording over which the state has one width.
struct state_segment {
  std::size_t first, last;
};

// The recording cut into one range per width, lowest first, by reading the widths
// the run recorded. An insertion is the only thing that changes a state's width
// during a run -- a System narrows only when a walk reconciles it to a recorded
// step -- so a width that grows between two rows is where one happened, and the
// record says so without being told.
//
// A width that SHRINKS is refused. That is the check this shape exists for, and
// it is the one an inferred boundary can still fail: the ranges have to partition
// a recording whose width only ever climbs, or a sweep is transposing something
// the run did not do.
template <class Record>
std::vector<state_segment> state_segments(std::span<const Record> rec) {
    if (rec.size() < 2) {
        util::stop("state_segments: no recorded steps to sweep");
    }
    std::vector<state_segment> ret;
    std::size_t first = 0;
    for (std::size_t k = 0; k + 1 < rec.size(); ++k) {
        const std::size_t here = rec[k].state.size();
        const std::size_t above = rec[k + 1].state.size();
        if (above < here) {
            util::stop("state_segments: the state is " +
                       util::to_string(static_cast<int>(above)) +
                       " wide at step " +
                       util::to_string(static_cast<int>(k + 1)) +
                       " against " + util::to_string(static_cast<int>(here)) +
                       " below it, and a run does not narrow");
        }
        if (above > here) {
            ret.push_back({first, k});
            first = k;
        }
    }
    ret.push_back({first, rec.size() - 1});
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
    const std::vector<state_segment> segments = state_segments(rec);
    if (segment >= segments.size()) {
        util::stop("state_at_segment: the recording has no such segment");
    }
    if (segment == 0) {
        be_at_step(system, rec, 0);
        start = 0;
        base.assign(system.ode_size(), 0.0);
        system.ode_state(base.begin());
        return rec[0].time;
    }
    // The state between an insertion and the step after it, which no record
    // holds. It is the insertion's own map at the state below it, so the width
    // is the one recorded at the step above -- checked by asking for it.
    start = segments[segment].first;
    if (start + 1 >= rec.size()) {
        util::stop("state_at_segment: the insertion after step " +
                   util::to_string(static_cast<int>(start)) +
                   " has no step above it in a recording of " +
                   util::to_string(static_cast<int>(rec.size())) +
                   ", so there is no width to check it against");
    }
    be_at_step(system, rec, start);
    system.inserted_state(rec[start].time, rec[start].state.begin(), base);
    util::check_length(base.size(), rec[start + 1].state.size());
    return rec[start].time;
}

// Carry lambda back over a recording whose state widened, one segment per width,
// highest first. At the foot of every segment but the lowest sits a widening: the
// System is narrowed across it and the map that widened it is transposed there,
// so the rows it carries reach `parameter_adjoint` by the same route a step's do.
//
// The widened state between a widening and the step after it is what no record
// holds, so it is rebuilt here. `extra_splits` names steps at which a segment
// stops and resumes; a split outside every segment's interior cuts nothing.
// Returns how many ranges were swept, which is not the segment count -- an empty
// lowest segment is swept zero times, and a split adds one.
template <class Solver>
std::size_t solve_adjoint_over_insertions(
    Solver& solver,
    std::span<const step_record<typename Solver::system_type>> rec,
    row_batch& lambda, row_batch& parameter_adjoint,
    const std::vector<std::size_t>& extra_splits = {}) {
    using scalar = active_scalar<double>;
    using record = step_record<typename Solver::system_type>;
    auto& system = solver.get_system_ref();
    const std::vector<state_segment> segments = state_segments(rec);

    // The state each insertion produced, which the sweep starting inside a
    // segment runs from and no record holds. It is the insertion's own map,
    // evaluated here at the passive scalar for its value and transposed below
    // for its rows -- one function, so the two cannot disagree.
    std::vector<record> with_insertions(rec.begin(), rec.end());
    for (std::size_t j = 1; j < segments.size(); ++j) {
        const std::size_t at = segments[j].first;
        be_at_step(system, rec, at);
        system.inserted_state(rec[at].time, rec[at].state.begin(),
                             with_insertions[at].state);
        // A boundary is a step whose width grows, so at + 1 is a step this
        // recording has and is the width the insertion has to reach.
        util::check_length(with_insertions[at].state.size(),
                           rec[at + 1].state.size());
    }
    const std::span<const record> sweep_rec{with_insertions.data(),
                                            with_insertions.size()};

    // One tape for every widening this walk crosses. Clearing it returns it to an
    // empty recording and keeps the capacity, where one built per widening
    // regrows it.
    // ⚠️ THE WIDTH ON EXIT IS A PROMISE, AND A THROW IS AN EXIT. The descent
    // below starts at the run's own width and narrows as it goes, so a sweep
    // abandoned in the highest segment leaves the System at its widest -- where
    // every caller's tail widens back from the lowest and reads the mismatch as a
    // length error one call later, naming neither this walk nor what refused.
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
            solver.solve_adjoint(sweep_rec, lambda, parameter_adjoint, cuts[c],
                                 upper);
            upper = cuts[c];
            ++swept;
        }
        // A widening at the first recorded step leaves the lowest segment with no
        // step in it, which is what a run from an empty state gives.
        if (segment.first < upper) {
            solver.solve_adjoint(sweep_rec, lambda, parameter_adjoint,
                                 segment.first, upper);
            ++swept;
        }
        if (j == 0) {
            break;
        }

        const std::size_t at = segments[j].first;
        be_at_step(system, rec, at);
        const double when = rec[at].time;
        auto insert = [&](auto& active_system,
                          typename std::vector<scalar>::const_iterator x,
                          std::vector<scalar>& y) -> void {
            active_system.inserted_state(when, x, y);
        };
        row_batch narrowed;
        state_and_parameter_adjoints(tape, system, rec[at].state, lambda,
                                     insert, narrowed, parameter_adjoint);
        lambda = std::move(narrowed);
    }
    return swept;
}

// Step `forward` over the recording's segments from `from_segment` on, at the
// sizes the run took, widening where the run widened. `first` is the recorded
// step the walk begins at, which the caller has already put the System on. The
// schedule is read off the recording, because a recording row is a schedule row
// plus its state.
//
// The step sizes are replayed rather than the times: a size differenced back out
// of two recorded times is not the size that was taken, since fl(fl(t + h) - t)
// is not h, and a walk that chose its own would be differentiating a controller
// the model does not contain.
template <class Solver, class Record>
void advance_over_insertions(
    Solver& forward, std::span<const Record> rec, std::size_t from_segment,
    std::size_t first) {
    const std::vector<state_segment> segments = state_segments(rec);
    if (from_segment >= segments.size()) {
        util::stop("advance_over_insertions: the recording has no such segment");
    }
    for (std::size_t j = from_segment; j < segments.size(); ++j) {
        // The first entry is the start no step reached, which is how a recorded
        // run reads back.
        std::vector<recorded_step> segment_steps{
            {forward.time(), std::numeric_limits<double>::quiet_NaN()}};
        for (std::size_t k = first + 1; k <= segments[j].last; ++k) {
            // A recording row is a schedule row plus its state, so the schedule
            // is read off it rather than built beside it.
            segment_steps.push_back(rec[k]);
        }
        if (segment_steps.size() > 1) {
            forward.advance_recorded(segment_steps);
        }
        if (j + 1 < segments.size()) {
            // The same map the sweep transposes, so a tangent traverses exactly
            // the function under test rather than a second spelling of it.
            auto& sys = forward.get_system_ref();
            using value_type = typename Solver::value_type;
            std::vector<value_type> before(sys.ode_size());
            sys.ode_state(before.begin());
            std::vector<value_type> after;
            sys.inserted_state(rec[segments[j].last].time, before.begin(), after);
            forward.set_state_from_system();
            first = segments[j].last;
        }
    }
}

}
}

#endif
