// -*-c++-*-
#ifndef ODELIA_QUADRATURE_HPP_
#define ODELIA_QUADRATURE_HPP_

#include <cstddef>

namespace odelia {
namespace quadrature {

// The weight each grid point carries under the trapezium rule, for a grid whose
// positions are passive.
//
// Positions arrive through a callable returning double, so a width cannot carry
// a derivative however the caller stores its grid. That is the property a
// transpose of a quadrature needs: where the abscissa is itself state the
// weights move with it and a reduction owes a term this does not compute.
//
// Interval-major, adding each width to both of its ends, because that is how the
// sum it transposes is associated. The caller scales by its own output adjoint
// and pushes the result through its own partials.
//
// `stop_after(k)` ends the interior walk after the interval reaching point `k`;
// the closing interval then runs from there to the last point, which is always
// taken. Both are the driver's, so a forward and its transpose cannot disagree
// about where a walk ended.
template <class Position, class StopAfter>
void trapezium_weights(std::size_t n_point, Position x, StopAfter stop_after,
                       double* w) {
  if (n_point < 2) {
    return;
  }
  std::size_t upper = 0;
  double x1 = x(0);
  for (std::size_t k = 1; k + 1 < n_point; ++k) {
    const double x0 = x(k);
    const double width = x0 - x1;
    w[upper] += width;
    w[k] += width;
    upper = k;
    x1 = x0;
    if (stop_after(k)) {
      break;
    }
  }
  const double closing = x(n_point - 1) - x1;
  w[upper] += closing;
  w[n_point - 1] += closing;
}

}
}

#endif
