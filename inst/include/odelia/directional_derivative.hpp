// -*-c++-*-
#ifndef ODELIA_DIRECTIONAL_DERIVATIVE_HPP_
#define ODELIA_DIRECTIONAL_DERIVATIVE_HPP_

// Forward-over-reverse directional derivative.
//
// Computes df/dx along one seeded direction EXACTLY -- the analytic derivative
// of the code, not a finite difference -- while keeping the result active on
// whatever outer reverse (adjoint) tape the scalar S already lives on. An
// enclosing reverse sweep therefore differentiates df/dx further, giving a mixed
// second derivative with no finite-difference step and no kink amplification.
//
// This is the tool for "a directional derivative that must itself take part in
// an outer reverse-mode gradient". The motivating case is plant's density-
// transport term dg/dh (the derivative of an individual's growth rate with
// respect to its height), which is evaluated inside a d/dtheta reverse sweep:
// finite-differencing dg/dh and then differentiating that stencil on the tape
// is unreliable (a growth clamp makes the stencil straddle a kink, and /eps
// amplifies it). See plant#39 and odelia#38.
//
// How the layers compose:
//   * If S is a reverse-active scalar (xad::AReal<double>), the working type is
//     xad::FReal<xad::AReal<double>> -- one forward (tangent) layer on top of the
//     reverse layer. The forward layer yields df/dx; because each component is an
//     AReal<double>, df/dx comes back as an AReal<double> still linked to the
//     outer tape (this is the nested tangent-over-adjoint case, odelia#35).
//   * If S is plain double, FReal<double> is an ordinary forward-mode number and
//     this is just an exact first derivative -- no outer tape involved.
//
// Only the code inside `f` runs at the tangent type; nothing else in the
// surrounding reverse computation is affected. The caller is responsible for
// lifting the other inputs `f` reads into the tangent layer with a zero tangent
// (they are constants with respect to the seeded direction) -- use constant().

#include <XAD/XAD.hpp>
#include <utility>

namespace odelia {
namespace ad {

// The tangent layer over an outer scalar S.
template <class S>
using tangent_of = xad::FReal<S>;

// Lift an outer-scalar value into the tangent layer. The value component keeps
// the outer scalar S unchanged (so its outer-tape derivatives are preserved --
// this is deliberately NOT xad::value(v), which would strip the reverse layer),
// and the forward tangent is set to `tangent`. Pass tangent = 1 for the single
// direction being differentiated and 0 for every other input.
template <class S>
tangent_of<S> seed(const S& v, double tangent) {
  tangent_of<S> f;
  value(f) = v;
  derivative(f) = tangent;
  return f;
}

// A value carried into the tangent layer with zero forward tangent: a constant
// with respect to the seeded direction, but still carrying its outer-tape
// derivatives. Use this to lift every input of `f` other than the direction.
template <class S>
tangent_of<S> constant(const S& v) { return seed(v, 0.0); }

// df/dx along x's own direction, computed by forward-over-reverse.
//
// `f` receives x lifted into the tangent layer with tangent 1, and must lift any
// other inputs it reads with constant() (tangent 0). It returns the result in the
// tangent layer. The forward tangent of that result is df/dx; it is returned as
// an S, still active on the outer reverse tape when S is a reverse-active scalar.
template <class S, class F>
S directional_derivative(const S& x, F&& f) {
  tangent_of<S> y = std::forward<F>(f)(seed(x, 1.0));
  return derivative(y);
}

} // namespace ad
} // namespace odelia

#endif
