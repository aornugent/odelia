// -*-c++-*-
#ifndef ODELIA_HERMITE_INTERPOLATOR_HPP_
#define ODELIA_HERMITE_INTERPOLATOR_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <odelia/ode_util.hpp>

namespace odelia {
namespace interpolator {

// A C1 piecewise-cubic interpolant built from a value AND a slope at each knot.
//
// The value and the slope come from one polynomial, so a caller that needs both
// gets a consistent pair: slope(u) is the exact derivative of what eval(u) returns.
//
// Each span reads only its own two knots, so moving one knot changes the
// interpolant only in the two spans that touch it.
//
// Knot positions are double; values and slopes carry the working scalar S. The two
// halves of a build are separate: set_nodes lays out the spans from the positions,
// and set_data fills the coefficients. A caller whose positions are fixed for a run
// calls set_nodes once and set_data per stage.
//
// eval and slope take either a double position or an active one. At an active
// position the value is read at its passive part and the query's own derivative is
// grafted on through the slope, so d(value)/d(u) is recorded.
template <typename S>
class hermite_interpolator {
public:
  // Knot positions, strictly ascending; at least two are needed (one span).
  // Discards any data already set.
  void set_nodes(const std::vector<double>& x_) {
    if (x_.size() < 2) util::stop("hermite_interpolator: need at least 2 knots");
    for (std::size_t i = 1; i < x_.size(); ++i) {
      if (!(x_[i] > x_[i - 1]))
        util::stop("hermite_interpolator: knots must be strictly ascending");
    }
    x = x_;
    const std::size_t ns = x.size() - 1;
    spans.assign(ns, Span());
    for (std::size_t k = 0; k < ns; ++k) {
      spans[k].x0 = x[k];
      spans[k].inv_h = 1.0 / (x[k + 1] - x[k]);
    }
    // An equally spaced knot set indexes by arithmetic instead of a search.
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
    initialised = false;
  }

  // Values and dy/dx at the nodes already set, one entry each per node.
  void set_data(const std::vector<S>& y_, const std::vector<S>& dydx_) {
    if (spans.empty()) util::stop("hermite_interpolator: no knots set");
    util::check_length(y_.size(), x.size());
    util::check_length(dydx_.size(), x.size());
    y = y_;
    m = dydx_;
    for (std::size_t k = 0; k < spans.size(); ++k) {
      const double h = x[k + 1] - x[k];
      const S a = y[k], b = y[k + 1];
      const S sa = m[k] * h, sb = m[k + 1] * h;
      Span& s = spans[k];
      s.y0 = a;
      s.c1 = sa;
      s.c2 = 3.0 * (b - a) - 2.0 * sa - sb;
      s.c3 = 2.0 * (a - b) + sa + sb;
    }
    initialised = true;
  }

  // Nodes and data in one call, for a caller that rebuilds both together.
  void init(const std::vector<double>& x_, const std::vector<S>& y_,
            const std::vector<S>& dydx_) {
    set_nodes(x_);
    set_data(y_, dydx_);
  }

  void clear() {
    x.clear(); y.clear(); m.clear(); spans.clear();
    inv_h0 = 0.0;
    uniform = false;
    initialised = false;
  }

  bool is_initialised() const { return initialised; }
  std::size_t size() const { return x.size(); }
  double min() const { return x.front(); }
  double max() const { return x.back(); }
  const std::vector<double>& knots() const { return x; }

  // Value at u. Outside the knot range the end span's line is extended (value and
  // slope of the nearest end), which keeps the read C1 across the boundary instead
  // of letting a cubic run away.
  template <typename U>
  S eval(const U& u) const {
    check_initialised();
    const double up = util::to_passive(u);
    if constexpr (std::is_same_v<U, double>) {
      return value_at(up);
    } else {
      return graft(value_at(up), slope_at(up), u, up);
    }
  }

  template <typename U>
  S operator()(const U& u) const { return eval(u); }

  // dy/du at u -- the exact derivative of the polynomial eval() uses, as a value.
  //
  // The position is read passively, and unlike eval() nothing grafts the query's
  // derivative back on, because d(slope)/d(u) is the curvature and no span carries
  // one. An active query is therefore refused rather than answered with a silent
  // zero: eval() and value_and_slope() are the readers that take one.
  template <typename U>
  S slope(const U& u) const {
    static_assert(std::is_same_v<U, double>,
                  "hermite_interpolator::slope reads the position at its value, so "
                  "an active query's derivative has nowhere to go and would come "
                  "back as exactly zero. Use eval() or value_and_slope(), which "
                  "graft it.");
    check_initialised();
    return slope_at(u);
  }

  // Both from one knot lookup and one span load, so a caller wanting the pair at
  // many positions pays one lookup each rather than two.
  template <typename U>
  void value_and_slope(const U& u, S& value, S& dydu) const {
    check_initialised();
    const double up = util::to_passive(u);
    if (up <= x.front()) {
      value = y.front() + m.front() * (up - x.front());
      dydu = m.front();
    } else if (up >= x.back()) {
      value = y.back() + m.back() * (up - x.back());
      dydu = m.back();
    } else {
      const Span& s = spans[span_of(up)];
      const double t = (up - s.x0) * s.inv_h;
      value = s.y0 + t * (s.c1 + t * (s.c2 + t * s.c3));
      dydu = (s.c1 + t * (2.0 * s.c2 + t * 3.0 * s.c3)) * s.inv_h;
    }
    if constexpr (!std::is_same_v<U, double>) {
      // The span is indexed at the passive position, so d(value)/d(u) is not
      // recorded by the read itself; the slope carries it. Without this a height
      // adjoint through the interpolant measures as exactly zero.
      value = graft(value, dydu, u, up);
    }
  }

private:
  // One span's whole polynomial, contiguous: a query touches a single cache line
  // rather than one per coefficient array.
  struct Span {
    double x0 = 0.0, inv_h = 0.0;
    S y0{}, c1{}, c2{}, c3{};
  };

  // The query's derivative, materialised while its operands are alive. A deduced
  // return type here would hand back an XAD expression template referencing the
  // temporaries of this return statement, which die on return.
  template <typename U>
  static S graft(const S& value, const S& dydu, const U& u, double up) {
    static_assert(std::is_constructible_v<S, U>,
                  "hermite_interpolator: reading at an active position needs the "
                  "knot values on the same scalar, so the derivative of the query "
                  "has somewhere to go -- an active position with S = double would "
                  "silently drop it.");
    return value + dydu * (u - up);
  }

  S value_at(double u) const {
    if (u <= x.front()) return y.front() + m.front() * (u - x.front());
    if (u >= x.back())  return y.back()  + m.back()  * (u - x.back());
    const Span& s = spans[span_of(u)];
    const double t = (u - s.x0) * s.inv_h;
    return s.y0 + t * (s.c1 + t * (s.c2 + t * s.c3));
  }

  S slope_at(double u) const {
    if (u <= x.front()) return m.front();
    if (u >= x.back())  return m.back();
    const Span& s = spans[span_of(u)];
    const double t = (u - s.x0) * s.inv_h;
    return (s.c1 + t * (2.0 * s.c2 + t * 3.0 * s.c3)) * s.inv_h;
  }

  std::size_t span_of(double u) const {
    const std::size_t ns = spans.size();
    if (uniform) {
      const std::size_t k = static_cast<std::size_t>((u - x.front()) * inv_h0);
      return k < ns ? k : ns - 1;
    }
    const std::size_t k =
        static_cast<std::size_t>(std::upper_bound(x.begin(), x.end(), u) - x.begin());
    return k > 0 ? k - 1 : 0;
  }

  void check_initialised() const {
    if (!initialised) util::stop("hermite_interpolator: not initialised");
  }

  std::vector<double> x;   // knot positions, contiguous for the search
  std::vector<S> y, m;     // knot values and slopes, as supplied
  std::vector<Span> spans;
  double inv_h0 = 0.0;
  bool uniform = false;
  bool initialised = false;
};

}
}

#endif
