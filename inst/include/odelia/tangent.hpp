// -*-c++-*-
#ifndef ODELIA_TANGENT_HPP_
#define ODELIA_TANGENT_HPP_

#include <XAD/XAD.hpp>
#include <type_traits>

// The forward-mode vocabulary, apart from the reverse-mode one because a
// consumer can want this and not that: a package with no tape reaches for the
// tangent scalar and must not be handed a name for a tape it does not have.

namespace odelia {
namespace ode {

// The scalar a forward-mode (tangent) pass runs on: one tangent layer above T,
// carrying a directional derivative and no tape. Nest it for a curvature -- a
// tangent of a tangent -- which is the only second-order scalar this family
// uses.
template <typename T = double>
using tangent_scalar = typename xad::fwd<T>::active_type;

// A scalar that carries a direction rather than an adjoint accumulator.
template <typename S>
concept CarriesDirection = xad::ExprTraits<S>::isForward;

// A scalar whose derivative is itself active, which is the only reason to build
// one: a second derivative. It decides how many corrections an implicit node
// records, and whether an expansion about the working point needs its
// second-order terms at all -- at first order they multiply two quantities that
// are zero, and cost tape that reads back nothing.
template <class S>
concept SecondOrder = !std::is_same_v<typename S::derivative_type, double>;

// The direction a forward pass carries in, and the derivative it carries out.
// Everything about them is a pass-through except which scalars they accept.
//
// That is the whole reason they are named. One accessor spells this AND an
// adjoint scalar's accumulator, so on an adjoint scalar the same statement seeds
// a slot no forward pass reads, or reads one no sweep has written -- and neither
// raises anything, because both are the accessor doing exactly what it says.
template <typename S>
  requires CarriesDirection<S>
void seed_direction(S& x, double direction) {
  xad::derivative(x) = direction;
}

// Returned by value, and declared rather than deduced. By value because the
// argument is routinely a temporary -- the kernel whose derivative is being read
// -- and a reference into one is a reference a caller can outlive. It costs
// nothing to copy: a tangent carries no tape, so a copy of one is bytes, where a
// copy of an adjoint value would record an operation.
template <typename S>
  requires CarriesDirection<S>
typename S::derivative_type derivative_along(const S& x) {
  return xad::derivative(x);
}

}
}

#endif
