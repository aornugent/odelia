// -*-c++-*-
#ifndef ODELIA_ODE_INTERFACE_HPP_
#define ODELIA_ODE_INTERFACE_HPP_

#include <cstddef>
#include <algorithm>
#include <span>
#include <array>
#include <vector>
#include <odelia/ode_util.hpp>
#include <odelia/tangent.hpp>

#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <XAD/XAD.hpp>

namespace odelia {
namespace ode {

// Type alias for state vectors based on System's value_type
template<typename System>
using state_type = std::vector<typename System::value_type>;

// The scalar a reverse-mode (adjoint) pass runs on: one adjoint layer above T. This is
// what a System is rebound to for a gradient, and the type its seeds and recorded values
// have. T is a parameter so the layer can sit on another active scalar; at the default
// it sits on double, which is the scalar an ordinary solve is differentiated from.
template <typename T = double>
using active_scalar = typename xad::adj<T>::active_type;

// The tape a reverse pass records on, reached from the scalar rather than named
// beside it. Naming a tape independently of its scalar is right at one
// derivative width and silently wrong at another -- both spellings compile, and
// the one that queries the wrong tape reports no tape active and stops nothing.
template <typename T = double>
using adjoint_tape = typename active_scalar<T>::tape_type;

// Every active value among the members handed in, whatever shape they arrive in:
// a scalar, a container of them, a pair, a pointer to one, or an object that
// answers for_each_active. A member the visitor cannot be called with is skipped,
// so a class lists what it holds rather than deciding which of them carry a
// derivative -- which is the judgment that makes forgetting one likely.
//
// ⚠️ A MEMBER IN A SHAPE NOT LISTED HERE IS SKIPPED, NOT REFUSED. An active scalar
// inside something this does not open is passed over in silence, and the count a
// caller checks afterwards is the only thing that says so.
template <class F, class T>
void visit_active(F& f, T& x) {
  if constexpr (requires { x.for_each_active(f); }) {
    x.for_each_active(f);
  } else if constexpr (requires { x.first; x.second; }) {
    visit_active(f, x.first);
    visit_active(f, x.second);
  } else if constexpr (requires { x.begin(); x.end(); }) {
    for (auto& element : x) { visit_active(f, element); }
  } else if constexpr (requires { *x; x == nullptr; }) {
    if (x != nullptr) { visit_active(f, *x); }
  } else if constexpr (std::invocable<F&, T&>) {
    f(x);
  }
}

template <class F, class A, class B, class... Rest>
void visit_active(F& f, A& a, B& b, Rest&... rest) {
  visit_active(f, a);
  visit_active(f, b, rest...);
}

// A direction carried INSIDE an adjoint: one recording of a reverse pass,
// differentiated along one direction. A mixed second derivative in every input
// and one direction costs a single recording this way, where a tangent above a
// tangent costs one pass per input. The other nesting is not available -- an
// no tangent scalar wraps an adjoint one -- so the direction is always the inner
// layer.
template <typename T = double>
using directional_adjoint_scalar = typename xad::fwd_adj<T>::active_type;

template <typename T = double>
using directional_adjoint_tape = typename xad::fwd_adj<T>::tape_type;

// Refused on any other scalar, for the reason seed_direction is: the accessors
// below spell the inner direction AND the outer accumulator, so on the wrong
// scalar the same statement seeds a slot nothing reads and nothing raises.
template <typename S>
concept CarriesDirectionUnderAdjoint =
    !xad::ExprTraits<S>::isForward &&
    xad::ExprTraits<typename S::derivative_type>::isForward;

// The direction the inner tangent carries. Seeding the outer layer is what the
// sweep does and is not this; the two differ by one accessor.
template <typename S>
  requires CarriesDirectionUnderAdjoint<S>
void seed_inner_direction(S& x, double direction) {
  xad::derivative(xad::value(x)) = direction;
}

// What an input's adjoint holds after the sweep: the pass's derivative in that
// input, differentiated along the seeded direction.
template <typename S>
  requires CarriesDirectionUnderAdjoint<S>
double directional_adjoint(const S& x) {
  return xad::derivative(xad::derivative(x));
}


// A System that carries its own clock. One that does not is time homogeneous,
// and the calls below hand it no time rather than refusing it.
template <typename T>
concept HasOdeTime = requires(const T& t) {
  { t.ode_time() } -> std::convertible_to<double>;
};

// A System that can hand back a copy of itself on scalar U. The rebound type must
// itself be a System on U: a rebind_from() returning the wrong scalar fails here
// rather than on the first arithmetic inside the caller.
//
// U is named rather than defaulted. Defaulting it to the System's own scalar asks
// whether a System can rebind to the scalar it already has, which is a different
// question from the one every caller means and is answered yes by types that
// cannot do what the caller needs.
template <typename S, typename U>
concept Rebindable = requires(const S& s) {
  { s.template rebind_from<U>() };
  requires std::same_as<typename decltype(s.template rebind_from<U>())::value_type, U>;
};

// The System type rebound to scalar U, i.e. decltype(system.rebind_from<U>()).
// When the System has no rebind_from() the type is not evaluated (a harmless
// placeholder is used instead), so a holder can still be class-instantiated for
// systems that will never rebind; the actual use is gated on the concept.
template <typename S, typename U, bool = Rebindable<S, U>>
struct rebound_system {
  using type = decltype(std::declval<const S>().template rebind_from<U>());
};
template <typename S, typename U>
struct rebound_system<S, U, false> {
  using type = S;
};

// What one rate evaluation solved for that its state does not determine: the
// results of the inner solves it ran, in the order it ran them. A System that
// solves for nothing declares nothing and this is empty.
//
// Only values of that kind belong here. Anything a rate evaluation can recompute
// from the state it was handed MUST be recomputed, because the transpose runs
// through it -- loading such a value as a recorded double severs the tape there
// and the sweep comes back wrong with every number finite.
struct no_solved_values {};

template <class System>
struct solved_values {
  using type = no_solved_values;
};
template <class System>
  requires requires { typename System::solved_values; }
struct solved_values<System> {
  using type = typename System::solved_values;
};
template <class System>
using solved_values_t = typename solved_values<System>::type;

// A System whose rate evaluation solves for something its state does not
// determine: the branch of an inner root-find, an early exit, the point an
// optimisation landed on. A pass re-running the model in order to tape it has to
// take the RUN'S result rather than solve again -- re-solving risks landing the
// other side of a branch, and what is then taped is a function the run never
// computed, with every number finite.
//
// So the run STORES what it solved for and a later pass LOADS it. Which of the two
// is happening is not a flag anyone keeps: a walk hands over the values, and
// whether it hands them over to be written or to be read is the constness of what
// it hands over.
//
// The extent is one rate evaluation, which is what `derivs` is. A walk that hands
// over nothing opens no extent, so a reload out of band cannot read a record and a
// record cannot complete a state it was not taken at.
template <typename System>
concept SolvesForValues =
  requires(System s, solved_values_t<System>& into,
           const solved_values_t<System>& from) {
    s.store_solved(into);
    s.load_solved(from);
    s.end_solved();
  };

// One step of a schedule: a time to stop at, and the size that reached it where
// something recorded it. Both in one object, because a replay adding sizes does
// not land where the run landed -- a run sets its last step into an interval to
// the interval's end rather than adding to it, and fl(t + (t1 - t)) is not t1.
// Two vectors side by side can also be paired across different runs; one cannot.
//
// A NaN size is a time with no size known, which is a grid point rather than a
// recorded step: step TO it. So one schedule type covers a grid a caller chose
// and a run a caller recorded, and each entry says which it is rather than a
// second container's emptiness saying it for all of them.
//
// The first entry is where the schedule starts, which no step reached: NaN.
struct recorded_step {
  double time;
  double step_size;
};

// One row of a recording: the schedule row a replay would take, the state the run
// held there, and the wider state an insertion made of it where one followed. A
// recording row IS a schedule row plus its state, so it derives rather than
// repeating the two fields -- written out separately, they were two structs
// differing by one member, and pairing a time from one container with a state
// from another was a thing that compiled.
//
// `inserted` is what the run reached between this row and the step above it, and
// it is empty at every row no insertion followed. Recorded rather than inferred:
// the forward pass knew it made an insertion here, and a walk that recovers that
// from a width which grew has to refuse a width which shrinks, where a recorded
// fact cannot be wrong. It is a field rather than a row of its own because an
// insertion shares its time with the step below it -- two rows at one time would
// move the schedule a replay reads and the step index a recorded stage is
// addressed by.
template <typename System>
struct step_record : recorded_step {
  state_type<System> state;
  state_type<System> inserted;

  // What this step's five stages solved for, in order. FIVE and not six: the sixth
  // rate evaluation a step makes is the one at the state it ends at, which
  // first-same-as-last hands the next step as its own first rates -- and a sweep
  // re-derives that one at the state it was handed rather than reading it. So there
  // is no slot for it, which is what makes "a walk cannot trust the first stage of a
  // recording it jumped into" structural instead of a warning.
  std::array<solved_values_t<System>, 5> solved;

  // What the step above this row ran from: the wider state where an insertion
  // followed, and this row's own where none did.
  const state_type<System>& ran_from() const {
    return inserted.empty() ? state : inserted;
  }
};

// A System whose state vector gains entries during a run does not declare a
// concept for it. Two members carry the whole of it, and they are as mandatory as
// ode_size() for a System a sweep is asked to walk, so they are called directly
// like it:
//
//   set_recorded_state(y, time)  -- be the shape this recorded time implies, then
//                                   take these values. A run loads into the shape
//                                   it already built; a replay does not know it,
//                                   and derives it from the schedule the run was
//                                   driven by.
//   apply_insertion(time, x, y)  -- the insertion as a map, the state below it in
//                                   and the whole wider state out, so it runs at
//                                   any scalar and the sweep can transpose it. It
//                                   leaves the System holding that wider state.
//
// The first is as mandatory as ode_size(): a System replayed from a recording has
// to be loadable from a recorded state, and for a width that never moves that is
// its ordinary load. The second is asked for only where the width changed, so a
// System that never widens is never asked for it -- ode::apply_insertion above is
// what a walk calls, and passing the state through is what an insertion is for
// such a System.
//
// An insertion whose TIME depends on the parameters is a different map: its
// adjoint carries a term through that time which nothing here computes. A System
// walked by the sweep asserts its insertions are scheduled, not triggered.

// Opt-in domain check. A system may declare
//
//   bool ode_state_valid(const state_type& y) const;
//
// and the adaptive stepper will reject any step landing on a state it refuses,
// shrinking and retrying instead of committing it. Systems that do not declare it
// are unaffected: state_valid() below takes its other branch, so nothing is
// called and nothing costs anything.
//
// The predicate is handed the state *vector* rather than reading the system. The
// stepper's final derivs() does leave the system sitting on y, so either would
// work, but a predicate over a vector is testable without constructing a system
// and is honest about what it is judging.
template <typename System, typename StateType>
concept ChecksState = requires(const System& s, const StateType& y) {
  { s.ode_state_valid(y) } -> std::convertible_to<bool>;
};

template <typename System, typename StateType>
bool state_valid(const System& system, const StateType& y) {
  if constexpr (ChecksState<System, StateType>) {
    return system.ode_state_valid(y);
  } else {
    return true;
  }
}

// The recursive interface. Each helper walks a container of elements, threading
// one iterator through them, and is constrained on the one member it calls with
// the iterator it was handed. Constraining the call rather than the element is
// what puts the diagnostic here: an element whose state moves through some other
// scalar's iterator -- a double-typed element reached with an active one -- fails
// the constraint at the call rather than a page of errors inside the loop.
template <typename ForwardIterator>
size_t ode_size(ForwardIterator first, ForwardIterator last) {
  size_t ret = 0;
  while (first != last) {
    ret += first->ode_size();
    ++first;
  }
  return ret;
}

template <typename ForwardIterator>
size_t aux_size(ForwardIterator first, ForwardIterator last) {
  size_t ret = 0;
  while (first != last) {
    ret += first->aux_size();
    ++first;
  }
  return ret;
}

template <typename ForwardIterator, typename It>
  requires requires(std::iter_value_t<ForwardIterator>& e, It it) {
    { e.set_ode_state(it) } -> std::same_as<It>;
  }
It set_ode_state(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = first->set_ode_state(it);
    ++first;
  }
  return it;
}

template <typename ForwardIterator, typename It>
  requires requires(std::iter_value_t<ForwardIterator>& e, It it) {
    { e.ode_state(it) } -> std::same_as<It>;
  }
It ode_state(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = first->ode_state(it);
    ++first;
  }
  return it;
}

template <typename ForwardIterator, typename It>
  requires requires(std::iter_value_t<ForwardIterator>& e, It it) {
    { e.ode_rates(it) } -> std::same_as<It>;
  }
It ode_rates(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = first->ode_rates(it);
    ++first;
  }
  return it;
}

template <typename ForwardIterator, typename It>
  requires requires(std::iter_value_t<ForwardIterator>& e, It it) {
    { e.ode_aux(it) } -> std::same_as<It>;
  }
It ode_aux(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = first->ode_aux(it);
    ++first;
  }
  return it;
}

template <typename ForwardIterator, typename It>
  requires requires(std::iter_value_t<ForwardIterator>& e, It it) {
    { e.set_ode_aux(it) } -> std::same_as<It>;
  }
It set_ode_aux(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = first->set_ode_aux(it);
    ++first;
  }
  return it;
}

template <typename T>
double ode_time(const T& obj) {
  if constexpr (HasOdeTime<T>) {
    return obj.ode_time();
  } else {
    return 0.0;
  }
}

namespace internal {
template <typename T, typename StateType>
void set_ode_state(T& obj, const StateType& y, double time) {
  if constexpr (HasOdeTime<T>) {
    obj.set_ode_state(y.begin(), time);
  } else {
    obj.set_ode_state(y.begin());
  }
}

}

// primarily for Ode_R - maybe remove
template <typename T, typename StateType>
void derivs(T& obj, const StateType& y, StateType& dydt,
            const double time) {

  internal::set_ode_state(obj, y, time);
  obj.ode_rates(dydt.begin());
}

// One rate evaluation's use of what it solves for. Opened before the state is
// loaded, because a System's own inner solves can happen inside the load, and
// closed however the evaluation leaves -- which is the whole reason it is a scope
// and not a pair of calls in two places.
//
// Store or load is decided by the CONSTNESS of what it was handed, so there is no
// mode to keep and nothing to hold a stale answer.
//
// `end_solved()` runs from a destructor, so it must not raise: it closes an extent
// and has nothing to fail at.
template <class System, class Values>
struct solved_scope {
  solved_scope(System& system, Values& values) : system_(system) {
    if constexpr (std::is_const_v<Values>) {
      system_.load_solved(values);
    } else {
      system_.store_solved(values);
    }
  }
  ~solved_scope() { system_.end_solved(); }
  solved_scope(const solved_scope&) = delete;
  solved_scope& operator=(const solved_scope&) = delete;

private:
  System& system_;
};

// One rate evaluation, handed what its inner solves are to write into, or what an
// earlier pass wrote for it. The two differ only in constness; a System that
// solves for nothing is never handed either.
template <typename T, typename StateType, typename Values>
  requires SolvesForValues<T> &&
           std::same_as<std::remove_const_t<Values>, solved_values_t<T>>
void derivs(T& obj, const StateType& y, StateType& dydt, const double time,
            Values& solved) {
  const solved_scope<T, Values> extent{obj, solved};
  internal::set_ode_state(obj, y, time);
  obj.ode_rates(dydt.begin());
}

// R interface functions - always use std::vector<double>
template <typename T>
std::vector<double> r_derivs(T& obj, const std::vector<double>& y, const double time) {
  std::vector<double> dydt(obj.ode_size());
  derivs(obj, y, dydt, time);
  return dydt;
}

// Two arities rather than one, because R binds each separately: a System with a
// clock is set at a time and one without is not offered one.
template <typename T>
  requires HasOdeTime<T>
void r_set_ode_state(T& obj, const std::vector<double>& y, double time) {
  util::check_length(y.size(), obj.ode_size());
  obj.set_ode_state(y.begin(), time);
}

template <typename T>
  requires (!HasOdeTime<T>)
void r_set_ode_state(T& obj, const std::vector<double>& y) {
  util::check_length(y.size(), obj.ode_size());
  obj.set_ode_state(y.begin());
}

template <typename T>
double r_ode_time(const T& obj) {
  return ode_time(obj);
}

template <typename T>
std::vector<double> r_ode_state(const T& obj) {
  std::vector<double> values(obj.ode_size());
  obj.ode_state(values.begin());
  return values;
}

// Mutable, unlike r_ode_state: a system's ode_rates may compute the rates for
// the state it currently holds rather than return a cached vector.
template <typename T>
std::vector<double> r_ode_rates(T& obj) {
  std::vector<double> dydt(obj.ode_size());
  obj.ode_rates(dydt.begin());
  return dydt;
}

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

// Apply the insertion the run took at `time`, from the state below it, and read
// back the state it produced: the insertion as a map, so it runs at any scalar
// and a sweep can transpose it.
//
// ⚠️ THE SYSTEM IS LEFT HOLDING THAT WIDER STATE, and no version of this leaves a
// widening System where it was: pushing the nodes is how the state is computed. So
// a walk rebinds again below this rather than sweeping the width below on what this
// ran on.
//
// A System whose width never changes inserts nothing, so the state passes
// through. That is not a fallback for a System that forgot to declare one -- it
// is what an insertion is for a width that does not move, and it is what lets a
// recording of such a System be walked without it implementing a map it is never
// asked for.
template <class System, class It>
void apply_insertion(System& system, double time, It x,
                     state_type<System>& out) {
  if constexpr (requires { system.apply_insertion(time, x, out); }) {
    system.apply_insertion(time, x, out);
  } else {
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = *x++;
    }
  }
}

template <typename T>
std::vector<double> r_ode_aux(const T& obj) {
  std::vector<double> dydt(obj.aux_size());
  obj.ode_aux(dydt.begin());
  return dydt;
}

template <typename T>
void r_set_ode_aux(T& obj, const std::vector<double>& aux) {
  util::check_length(aux.size(), obj.aux_size());
  obj.set_ode_aux(aux.begin());
}

}
}

#endif
