// -*-c++-*-
#ifndef ODELIA_ODE_INTERFACE_HPP_
#define ODELIA_ODE_INTERFACE_HPP_

#include <cstddef>
#include <vector>
#include <odelia/ode_util.hpp>

#include <concepts>
#include <iterator>
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

// By default, we assume that systems are time homogeneous
template <typename T>
class needs_time {
  typedef char true_type;
  typedef long false_type;
  template <typename C> static true_type test(decltype(&C::ode_time)) ;
  template <typename C> static false_type test(...);
public:
  enum { value = sizeof(test<T>(0)) == sizeof(true_type) };
};

// A System that records the node positions its adaptive solve chose, so a later
// pass runs on a schedule that no longer moves and the active scalar propagates
// through it. The field is then recomputed at those fixed positions, and its
// derivative flows. The hook is detected at compile time; a System that records
// nothing is unaffected and pays nothing.
template <typename System>
concept RecordsSteps = requires(System s) {
  s.record_ode_step();       // per accepted ODE step: commit the node positions
};

// And a System that keeps a field value per RK stage as well, reading it back by
// stage index instead of recomputing it. Those values are reused as fixed doubles,
// so the derivative through the field is zero rather than flowing;
// has_recorded_field() reports which of the two depths applies.
//
// Separate from RecordsSteps because recording the steps and rebuilding the field
// is the ordinary case. Asking for it as one concept made such a System declare
// three members it did not mean, and an empty body is how a hook stops being a
// contract and becomes a formality nobody reads -- including, eventually, the
// engine that was supposed to call it.
template <typename System>
concept ReplaysField =
  requires(System s, int stage,
           typename std::vector<typename System::value_type>::const_iterator in) {
    s.record_stage(stage);     // per RK stage: record this stage's field value
    s.replay_step();           // per step on the replay pass: restore its record
    { s.has_recorded_field() } -> std::convertible_to<bool>;
    s.set_ode_state(in, stage);  // load state against a recorded stage, not a time
  };

// A System whose state vector gains entries at times the RUN schedules. The
// entries are the model's -- what they mean, and how many one widening adds, is
// never read here; `widening` is opaque and only ever handed back.
//
// Three members because three different things happen to the state. widen() and
// narrow() move the System itself, and narrow() is not derivable from a width:
// which entries to drop, and what to rebuild afterwards, is the model's. Between
// them they must round-trip, and in reverse order across several widenings,
// which no signature here can say.
//
// widened_state() is the map alone -- the narrow state in, the wide state out,
// nothing else rebuilt -- so it can be evaluated at an active scalar and taped.
// It loads the state it is given, which is why nothing here names a state
// loader. It leaves the System holding what it added, so a caller evaluating it
// more than once narrows between calls.
//
// A widening whose TIME depends on the parameters is a different map: its
// adjoint carries a term through that time which nothing here computes. A System
// satisfying this asserts its widenings are scheduled, not triggered.
template <typename System>
concept WidensState =
  requires(System s, const typename System::widening& w, double time,
           typename std::vector<typename System::value_type>::const_iterator in,
           std::vector<typename System::value_type>& out) {
    typename System::widening;
    s.widen(w);
    s.narrow(w);
    s.widened_state(w, time, in, out);
  };

// A System that carries the transpose of its own rate evaluation, for several
// rate adjoints at once: ode_rates_adjoint_batched takes the adjoints of dydt to
// the adjoints of y, recording whatever the transpose is built on once and
// sweeping it per seed. Where the recording is a model evaluation and the sweep
// is arithmetic -- which is the case for any System whose rates are a solve -- a
// seed past the first is nearly free, and a caller wanting several rows of one
// trajectory pays for one.
//
// set_ode_state_for_adjoint puts the System where that transpose is taken from,
// doing only what the transpose does not redo -- a System that rebuilds its
// field inside its own recording leaves it here, and one that does not builds it
// here, and neither can be told which the other is. The aux members carry a
// stage's operating point across, so the transpose reads back the point the
// rates were evaluated at.
//
// `twin` is the System at the adjoint scalar, owned by the caller and handed to
// every stage. The System writes its own values into it before each recording:
// what a recording writes into it points at slots the next one clears, so a twin
// used as it arrives carries a cleared recording's slots into this one and the
// sweep comes back wrong -- one seed's rows exact and another's not, with
// nothing raised.
template <class System, class Twin>
concept AdjointRates =
  requires(System s, double time,
           const std::vector<std::vector<typename System::value_type>>& rates_in,
           std::vector<std::vector<typename System::value_type>>& state_out,
           std::vector<std::vector<double>>& parameter_adjoint, Twin& twin,
           typename std::vector<typename System::value_type>::const_iterator in,
           typename std::vector<typename System::value_type>::iterator out) {
    { s.set_ode_state_for_adjoint(in, time) } -> std::same_as<decltype(in)>;
    { s.ode_rates_adjoint_batched(rates_in, state_out, parameter_adjoint, twin) }
      -> std::same_as<void>;
    { s.aux_size() } -> std::convertible_to<size_t>;
    { s.ode_aux(out) } -> std::same_as<decltype(out)>;
    { s.set_ode_aux(in) } -> std::same_as<decltype(in)>;
  };

// Opt-in domain check (#55). A system may declare
//
//   bool ode_state_valid(const state_type& y) const;
//
// and the adaptive stepper will reject any step landing on a state it refuses,
// shrinking and retrying instead of committing it. Systems that do not declare it
// are unaffected: state_valid() below resolves to the constant-true overload, so
// nothing is called and nothing costs anything.
//
// The predicate is handed the state *vector* rather than reading the system. The
// stepper's final derivs() does leave the system sitting on y, so either would
// work, but a predicate over a vector is testable without constructing a system
// and is honest about what it is judging.
template <typename System>
class has_state_check {
  typedef char true_type;
  typedef long false_type;
  template <typename C> static true_type test(decltype(&C::ode_state_valid)) ;
  template <typename C> static false_type test(...);
public:
  enum { value = sizeof(test<System>(0)) == sizeof(true_type) };
};

template <typename System, typename StateType>
typename std::enable_if<has_state_check<System>::value, bool>::type
state_valid(const System& system, const StateType& y) {
  return system.ode_state_valid(y);
}

template <typename System, typename StateType>
typename std::enable_if<!has_state_check<System>::value, bool>::type
state_valid(const System& /* system */, const StateType& /* y */) {
  return true;
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
typename std::enable_if<needs_time<T>::value, double>::type
ode_time(const T& obj) {
  return obj.ode_time();
}

template <typename T>
typename std::enable_if<!needs_time<T>::value, double>::type
ode_time(const T& /* obj */) {
  return 0.0;
}

namespace internal {
template <typename T, typename StateType>
typename std::enable_if<needs_time<T>::value, void>::type
set_ode_state(T& obj, const StateType& y, double time) {
  obj.set_ode_state(y.begin(), time);
}

template <typename T, typename StateType>
typename std::enable_if<!needs_time<T>::value, void>::type
set_ode_state(T& obj, const StateType& y, double /* time */) {
  obj.set_ode_state(y.begin());
}

template <typename T, typename StateType>
  requires ReplaysField<T>
void set_ode_state(T& obj, const StateType& y, int index) {
  obj.set_ode_state(y.begin(), index);
}
}

// primarily for Ode_R - maybe remove
template <typename T, typename StateType>
void derivs(T& obj, const StateType& y, StateType& dydt,
            const double time) {

  internal::set_ode_state(obj, y, time);
  obj.ode_rates(dydt.begin());
}

// ODE stepping. A System that has recorded field values reads the field for this RK
// stage by index; otherwise it sets state at the current time and recomputes (the
// second branch also covers every System that keeps no field). The choice compiles
// away for a System that does not replay one.
template <typename T, typename StateType>
void derivs(T& obj, const StateType& y, StateType& dydt,
            const double time, const int index) {
  if constexpr (ReplaysField<T>) {
    if (obj.has_recorded_field()) {
      internal::set_ode_state(obj, y, index);
    } else {
      internal::set_ode_state(obj, y, time);
    }
  } else {
    internal::set_ode_state(obj, y, time);
  }
  obj.ode_rates(dydt.begin());
}

// R interface functions - always use std::vector<double>
template <typename T>
std::vector<double> r_derivs(T& obj, const std::vector<double>& y, const double time) {
  std::vector<double> dydt(obj.ode_size());
  derivs(obj, y, dydt, time);
  return dydt;
}

template <typename T>
typename std::enable_if<needs_time<T>::value, void>::type
r_set_ode_state(T& obj, const std::vector<double>& y, double time) {
  util::check_length(y.size(), obj.ode_size());
  obj.set_ode_state(y.begin(), time);
}

template <typename T>
typename std::enable_if<!needs_time<T>::value, void>::type
r_set_ode_state(T& obj, const std::vector<double>& y) {
  util::check_length(y.size(), obj.ode_size());
  obj.set_ode_state(y.begin());
}

template <typename T>
typename std::enable_if<needs_time<T>::value, double>::type
r_ode_time(const T& obj) {
  return obj.ode_time();
}

template <typename T>
typename std::enable_if<!needs_time<T>::value, double>::type
r_ode_time(const T& /* obj */) {
  return 0.0;
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
