// -*-c++-*-
#ifndef ODELIA_HERMITE_INTERPOLATOR_HPP_
#define ODELIA_HERMITE_INTERPOLATOR_HPP_

#include <odelia/ode_util.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace odelia {
namespace interpolator {

// A C1 piecewise-cubic interpolant built from a value AND a slope at each knot.
//
// The value and the slope come from one polynomial, so a caller that needs both
// gets a consistent pair: slope(u) is the exact derivative of what eval(u)
// returns. An interpolant fitted to values alone offers no such guarantee -- its
// analytic tangent is whatever the fit happened to produce, and where the target
// has a curvature break between knots that tangent is wrong by an amount the
// value error does not reveal.
//
// Each span reads only its own two knots. That locality is what makes an adjoint
// cheap: a query reaches back to two knots rather than to the whole knot set, as
// a C2 fit does through its band solve. It also means moving a knot changes the
// interpolant only in the two spans that touch it, and that value and slope stay
// continuous across the change, since both are pinned at the shared knot.
//
// Knot positions are double; values and slopes carry the working scalar S. A
// query is indexed at the passive position, so d(value)/d(u) is not recorded --
// slope(u) returns it explicitly to a caller that wants it.
template <typename S>
class hermite_interpolator {
public:
  // Knots, their values, and dy/dx at each. Positions must be strictly
  // ascending; at least two are needed (one span).
  void init(const std::vector<double>& x_, const std::vector<S>& y_,
            const std::vector<S>& dydx_) {
    util::check_length(y_.size(), x_.size());
    util::check_length(dydx_.size(), x_.size());
    if (x_.size() < 2) util::stop("hermite_interpolator: need at least 2 knots");
    for (std::size_t i = 1; i < x_.size(); ++i) {
      if (!(x_[i] > x_[i - 1]))
        util::stop("hermite_interpolator: knots must be strictly ascending");
    }
    x = x_;
    y = y_;
    m = dydx_;
    rebuild();
  }

  void clear() {
    x.clear(); y.clear(); m.clear(); spans.clear();
    uniform = false;
    active = false;
  }

  bool is_active() const { return active; }
  std::size_t size() const { return x.size(); }
  double min() const { return x.front(); }
  double max() const { return x.back(); }
  const std::vector<double>& knots() const { return x; }

  // Value at u. Outside the knot range the end span's line is extended (value
  // and slope of the nearest end), which keeps the read C1 across the boundary
  // instead of letting a cubic run away.
  S eval(double u) const {
    check_active();
    if (u <= x.front()) return y.front() + m.front() * (u - x.front());
    if (u >= x.back())  return y.back()  + m.back()  * (u - x.back());
    const Span& s = spans[span_of(u)];
    const double t = (u - s.x0) * s.inv_h;
    return s.y0 + t * (s.c1 + t * (s.c2 + t * s.c3));
  }

  S operator()(double u) const { return eval(u); }

  // dy/du at u -- the exact derivative of the polynomial eval() uses. (odelia's
  // older value-fitted Interpolator spells the same operation `deriv`.)
  S slope(double u) const {
    check_active();
    if (u <= x.front()) return m.front();
    if (u >= x.back())  return m.back();
    const Span& s = spans[span_of(u)];
    const double t = (u - s.x0) * s.inv_h;
    return (s.c1 + t * (2.0 * s.c2 + t * 3.0 * s.c3)) * s.inv_h;
  }

  // Both from one knot lookup and one span load. A crown integral wants the pair
  // at every quadrature point, so this halves that work.
  void value_and_slope(double u, S& value, S& dydu) const {
    check_active();
    if (u <= x.front()) { value = y.front() + m.front() * (u - x.front()); dydu = m.front(); return; }
    if (u >= x.back())  { value = y.back()  + m.back()  * (u - x.back());  dydu = m.back();  return; }
    const Span& s = spans[span_of(u)];
    const double t = (u - s.x0) * s.inv_h;
    value = s.y0 + t * (s.c1 + t * (s.c2 + t * s.c3));
    dydu  = (s.c1 + t * (2.0 * s.c2 + t * 3.0 * s.c3)) * s.inv_h;
  }

private:
  // One span's whole polynomial, contiguous: a query touches a single cache line
  // rather than one per coefficient array.
  struct Span {
    double x0, inv_h;
    S y0, c1, c2, c3;
  };

  void rebuild() {
    const std::size_t ns = x.size() - 1;
    spans.resize(ns);
    for (std::size_t k = 0; k < ns; ++k) {
      const double h = x[k + 1] - x[k];
      const S a = y[k], b = y[k + 1];
      const S sa = m[k] * h, sb = m[k + 1] * h;
      Span& s = spans[k];
      s.x0 = x[k];
      s.inv_h = 1.0 / h;
      s.y0 = a;
      s.c1 = sa;
      s.c2 = 3.0 * (b - a) - 2.0 * sa - sb;
      s.c3 = 2.0 * (a - b) + sa + sb;
    }
    // An equally spaced knot set indexes by arithmetic instead of a search.
    // Cohort heights are not equally spaced; a fixed environment's are.
    uniform = false;
    if (ns > 1) {
      const double h0 = x[1] - x[0];
      const double tol = 1e-12 * (x.back() - x.front());
      uniform = true;
      for (std::size_t k = 1; k < ns; ++k) {
        if (std::abs((x[k + 1] - x[k]) - h0) > tol) { uniform = false; break; }
      }
      if (uniform) inv_h0 = 1.0 / h0;
    }
    active = true;
  }

  std::size_t span_of(double u) const {
    const std::size_t ns = spans.size();
    if (uniform) {
      const std::size_t k =
          static_cast<std::size_t>((u - x.front()) * inv_h0);
      return k < ns ? k : ns - 1;
    }
    const std::size_t k =
        static_cast<std::size_t>(std::upper_bound(x.begin(), x.end(), u) - x.begin());
    return k > 0 ? k - 1 : 0;
  }

  void check_active() const {
    if (!active) util::stop("hermite_interpolator: not initialised");
  }

  std::vector<double> x;      // knot positions, contiguous for the search
  std::vector<S> y, m;        // knot values and slopes, as supplied
  std::vector<Span> spans;
  double inv_h0 = 0.0;
  bool uniform = false;
  bool active = false;
};

}  // namespace interpolator
}  // namespace odelia

#endif
