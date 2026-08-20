// -*-c++-*-
#ifndef ODELIA_ODE_INTERFACE_HPP_
#define ODELIA_ODE_INTERFACE_HPP_

#include <cstddef>
#include <vector>
#include <odelia/ode_util.hpp>

#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>
#include <XAD/XAD.hpp>

namespace odelia {
namespace ode {

// Type alias for state vectors based on System's value_type
template<typename System>
using state_type = std::vector<typename System::value_type>;

// The scalar a reverse-mode (adjoint) pass runs on: one adjoint layer above T. This is
// what a System is lifted to for a gradient, and the type its seeds and recorded values
// have. T is a parameter so the layer can sit on another active scalar; at the default
// it sits on double, which is the scalar an ordinary solve is differentiated from.
template <typename T = double>
using active_scalar = typename xad::adj<T>::active_type;

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

// Where in a run one rate evaluation sits: the accepted step it belongs to, and
// which of that step's stages it is. One value rather than two arguments, because
// the two halves are one address -- and because a type of its own is what stops it
// being taken for the time, which one bare number beside another does not.
//
// A step makes six rate evaluations: its stages 1 to 5, and one at the state it ends
// at, which first-same-as-last hands the NEXT step as that step's first rates. So the
// sixth is addressed as the next step's stage 0, and a pass re-running the schedule
// forward reads it back where the run wrote it.
//
// A reverse recording does not: it re-derives stage 0 at the state it was handed and
// asks for nothing there. Between two steps the run can widen its state, and it then
// takes its first rates at a state no record holds -- so stage 0 is the one address a
// walk that jumps into the middle of a recording cannot trust.
struct recorded_stage {
  std::size_t step;
  int stage;
};

// And a System whose state does not determine it. Between one rate evaluation and
// the next this System makes choices the state leaves open -- the nodes a
// refinement placed, the branch an inner solve took, an early exit -- and a pass
// that re-runs the model in order to tape it has to make the RUN'S choices rather
// than its own. Re-deriving them risks a different discretisation, and what is then
// taped is a function the run never computed, with every number finite.
//
// So the run records them against the evaluation that made them, and every pass
// re-running the model loads them with the state. That is the same requirement
// set_recorded_state exists for one level up, and it is spelled as a loader for the
// same reason: what completes a state belongs with the state.
//
// The address reaches a System only from a walk that is stepping, so a reload out
// of band cannot read a record, and a record cannot complete a state it was not
// taken at.
template <typename System>
concept RecordsChoices =
  requires(System s, double time, recorded_stage at,
           typename std::vector<typename System::value_type>::const_iterator in) {
    s.set_ode_state(in, time, at);
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

// One insertion the run made: what it added, the recorded step it followed, and
// that step's recorded time. `what` is the model's own and is never read here.
//
// The time sits beside the widening rather than being looked up beside it because
// the two are one fact -- the run inserts at the step it has just recorded. It is
// derived from the record rather than declared, so it cannot disagree with the
// step it belongs to.
template <typename Widening>
struct recorded_insertion {
  Widening what;
  std::size_t after_step;
  double time;
};

// A System whose state vector gains entries at times the RUN schedules. The
// entries are the model's -- what they mean, and how many one widening adds, is
// never read here; `widening` is opaque and only ever handed back.
//
// Two members, and both are loads. There is no widen() and no narrow(): a walk
// says which recorded step to be at and the System reconciles itself to it, so
// nothing sequences insertions, nothing undoes them in reverse, and there is no
// claim that one is the other's inverse for a round trip to have to check.
//
// set_recorded_state() puts the System on the state the run recorded at a step,
// carrying the insertions that had happened by then. It is one call and not a
// load beside a widening because a run carries more than its state vector: what
// an inserted entry is stamped with, and a quantity the rates evaluate twice,
// are the model's and are in neither the state nor the width. Reconciling to a
// target rather than stepping from a cursor is what makes it idempotent -- being
// at a step twice is being there once.
//
// widened_state() is the insertion alone as a map -- the state before it in, the
// whole widened state out, nothing else rebuilt -- so it can be evaluated at an
// active scalar and taped. It is the one the sweep transposes.
//
// A widening whose TIME depends on the parameters is a different map: its
// adjoint carries a term through that time which nothing here computes. A System
// satisfying this asserts its insertions are scheduled, not triggered.
template <typename System>
concept WidensState =
  requires(System s, const recorded_insertion<typename System::widening>& one,
           const std::vector<recorded_insertion<typename System::widening>>& all,
           std::size_t n, double time,
           const std::vector<typename System::value_type>& y,
           typename std::vector<typename System::value_type>::const_iterator in,
           std::vector<typename System::value_type>& out) {
    typename System::widening;
    s.set_recorded_state(y, time, all, n);
    s.widened_state(one, in, out);
  };

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

// The same load, with the address of the choices the run made at this evaluation.
// A System that records none loads without it.
template <typename T, typename StateType>
void set_ode_state(T& obj, const StateType& y, double time, recorded_stage at) {
  if constexpr (RecordsChoices<T>) {
    obj.set_ode_state(y.begin(), time, at);
  } else {
    set_ode_state(obj, y, time);
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

// One rate evaluation of a step, at the address the run recorded its choices
// against. A System that records none takes the call above and the address costs
// it nothing.
template <typename T, typename StateType>
void derivs(T& obj, const StateType& y, StateType& dydt, const double time,
            recorded_stage at) {
  internal::set_ode_state(obj, y, time, at);
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
