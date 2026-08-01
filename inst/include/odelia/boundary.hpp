// -*-c++-*-
#ifndef ODELIA_BOUNDARY_HPP_
#define ODELIA_BOUNDARY_HPP_

#include <odelia/ode_util.hpp>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace odelia {

// A boundary is one contiguous run of values an element publishes under a name.
// A Channel is that name: it declares which of four directions the run has --
// a width, a read, a write, an adjoint write -- and forwards each to the member
// that implements it. A run with only three of them names only three.

// Each direction is constrained on the one call it makes, with the iterator it was
// handed, so an element that cannot move its run through this iterator fails here
// rather than inside the loop.
template <typename Channel, typename Element>
concept Declares = requires(const Element& e) {
  { Channel::width(e) } -> std::convertible_to<std::size_t>;
};

template <typename Channel, typename Element, typename It>
concept Reads = requires(const Element& e, It it) {
  { Channel::read(e, it) } -> std::same_as<It>;
};

template <typename Channel, typename Element, typename It>
concept Writes = requires(Element& e, It it) {
  { Channel::write(e, it) } -> std::same_as<It>;
};

// The adjoint of the write: the same run, in the same order, accumulating instead of
// assigning.
template <typename Channel, typename Element, typename It>
concept WritesAdjoint = requires(Element& e, It it) {
  { Channel::write_adjoint(e, it) } -> std::same_as<It>;
};

// Output iterator that counts assignments and keeps none of them, so a read can be
// asked how wide it is without a buffer to read into.
class width_counter {
public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = void;

  struct sink {
    template <typename T> const sink& operator=(const T&) const { return *this; }
  };

  sink operator*() const { return sink(); }
  width_counter& operator++() { ++n_; return *this; }
  width_counter operator++(int) { width_counter prev(*this); ++n_; return prev; }
  std::size_t count() const { return n_; }

private:
  std::size_t n_ = 0;
};

// The width a read moves. Derived, so a channel whose width is another channel's --
// a rate run is as wide as the state run it differentiates -- declares no second
// width to disagree with the first.
template <typename Channel, typename Element>
  requires Reads<Channel, Element, width_counter>
std::size_t read_width(const Element& e) {
  return Channel::read(e, width_counter()).count();
}

// A declared width and the width its read moves must agree. Runtime, because both are
// data: a cohort count, the soil layers holding root mass.
template <typename Channel, typename Element>
  requires Reads<Channel, Element, width_counter>
void check_width(const Element& e) {
  if constexpr (Declares<Channel, Element>) {
    util::check_length(read_width<Channel>(e), Channel::width(e));
  }
}

// The same four directions over a container of elements, threading one iterator
// through them, in the element order the container gives.
template <typename Channel, typename ForwardIterator>
  requires Declares<Channel, std::iter_value_t<ForwardIterator>>
std::size_t width(ForwardIterator first, ForwardIterator last) {
  std::size_t ret = 0;
  while (first != last) {
    ret += Channel::width(*first);
    ++first;
  }
  return ret;
}

template <typename Channel, typename ForwardIterator, typename It>
  requires Reads<Channel, std::iter_value_t<ForwardIterator>, It>
It read(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = Channel::read(*first, it);
    ++first;
  }
  return it;
}

template <typename Channel, typename ForwardIterator, typename It>
  requires Writes<Channel, std::iter_value_t<ForwardIterator>, It>
It write(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = Channel::write(*first, it);
    ++first;
  }
  return it;
}

template <typename Channel, typename ForwardIterator, typename It>
  requires WritesAdjoint<Channel, std::iter_value_t<ForwardIterator>, It>
It write_adjoint(ForwardIterator first, ForwardIterator last, It it) {
  while (first != last) {
    it = Channel::write_adjoint(*first, it);
    ++first;
  }
  return it;
}

namespace ode {

// The ODE boundary, as the runs it is made of. The state run has all four
// directions where an element supplies the fourth; the rate run is a read whose
// width is the state run's; the aux run has a width, a read and a write.
struct state_channel {
  template <typename E> static std::size_t width(const E& e) { return e.ode_size(); }
  template <typename E, typename It> static It read(const E& e, It it) {
    return e.ode_state(it);
  }
  template <typename E, typename It> static It write(E& e, It it) {
    return e.set_ode_state(it);
  }
  template <typename E, typename It> static It write_adjoint(E& e, It it) {
    return e.set_ode_state_adjoint(it);
  }
};

struct rate_channel {
  template <typename E, typename It> static It read(const E& e, It it) {
    return e.ode_rates(it);
  }
};

struct aux_channel {
  template <typename E> static std::size_t width(const E& e) { return e.aux_size(); }
  template <typename E, typename It> static It read(const E& e, It it) {
    return e.ode_aux(it);
  }
  template <typename E, typename It> static It write(E& e, It it) {
    return e.set_ode_aux(it);
  }
};

}
}

#endif
