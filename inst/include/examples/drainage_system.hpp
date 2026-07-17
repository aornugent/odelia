#ifndef DRAINAGE_SYSTEM_HPP_
#define DRAINAGE_SYSTEM_HPP_

#include <odelia/mri.hpp>
#include <vector>
#include <cmath>

// A stiff multirate demonstrator shaped like the TF24 soil column: L fast layers
// that drain by a power law (a stiff, wet-end nonlinearity with a closed-form
// recession) and refill gently toward a slow signal, coupled to M slow modes that
// track the layer mean. The drainage is the stiff part; splitting integrates it
// exactly (analytic_flow) and leaves ROS34PW2 only the gentle refill, so the same
// model can be run with and without splitting to measure what an analytic-flow
// exposure buys as the drainage stiffness grows.
//
// Fast:  u_l' = -c u_l^p            (drainage, stiff; recession u_l = [u_l^{1-p} + (p-1) c t]^{-1/(p-1)})
//              + r (g - u_l)        (gentle refill toward the slow signal g)
// Slow:  x_j' = w_j (mean(u) - x_j)
// g = mean(x).
template <typename T = double>
class DrainageSystem {
public:
  using value_type = T;
  using vec = std::vector<T>;

  DrainageSystem(double drain_c, int n_fast_, int n_slow_)
    : c(drain_c), p(16.0), r(0.3), n_fast(n_fast_), n_slow(n_slow_),
      omega(n_slow_), u(n_fast_), x(n_slow_), t0(0.0), time(0.0) {
    for (int j = 0; j < n_slow; ++j)
      omega[j] = 0.02 + 0.18 * (n_slow > 1 ? double(j) / (n_slow - 1) : 0.0);
    reset();
  }

  size_t fast_size() const { return (size_t)n_fast; }
  size_t slow_size() const { return (size_t)n_slow; }
  size_t coupling_size() const { return 1; }
  size_t ode_size() const { return fast_size() + slow_size(); }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  void aggregate(const vec& xs, vec& g) const {
    T s = T(0.0);
    for (const auto& xj : xs) s += xj;
    g[0] = s / double(n_slow);
  }

  void fast_rates(const vec& us, const vec& g, vec& du) const {
    for (int l = 0; l < n_fast; ++l)
      du[l] = -drainage(us[l]) + r * (g[0] - us[l]);
  }

  void slow_rates(const vec& xs, const vec& us, vec& dx) const {
    T m = T(0.0);
    for (const auto& ul : us) m += ul;
    m /= double(n_fast);
    for (int j = 0; j < n_slow; ++j) dx[j] = omega[j] * (m - xs[j]);
  }

  // Split hooks (double): the exact drainage recession, and the gentle remainder.
  void analytic_flow(std::vector<double>& us, double dt) const {
    for (auto& ul : us) {
      if (ul <= 0.0) continue;
      const double base = std::pow(ul, 1.0 - p) + (p - 1.0) * c * dt;
      ul = std::pow(base, -1.0 / (p - 1.0));
    }
  }
  void residual_rhs(const std::vector<double>& us, const std::vector<double>& g,
                    std::vector<double>& du) const {
    for (int l = 0; l < n_fast; ++l) du[l] = r * (g[0] - us[l]);
  }

  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    time = time_;
    for (auto& ul : u) ul = *it++;
    for (auto& xj : x) xj = *it++;
    return it;
  }
  template <typename Iterator>
  Iterator ode_state(Iterator it) const {
    for (const auto& ul : u) *it++ = ul;
    for (const auto& xj : x) *it++ = xj;
    return it;
  }
  template <typename Iterator>
  Iterator ode_rates(Iterator it) const {
    vec g(1), du(n_fast), dx(n_slow);
    aggregate(x, g);
    fast_rates(u, g, du);
    slow_rates(x, u, dx);
    for (const auto& d : du) *it++ = d;
    for (const auto& d : dx) *it++ = d;
    return it;
  }

  void reset() {
    for (auto& ul : u) ul = T(0.9);   // wet: drainage active
    for (int j = 0; j < n_slow; ++j) x[j] = T(0.5 + 0.2 * std::cos(3.14159265358979 * j / n_slow));
    time = t0;
  }

private:
  T drainage(const T& ul) const { using std::pow; return c * pow(ul, p); }

  double c, p, r;
  int n_fast, n_slow;
  std::vector<double> omega;
  vec u, x;
  double t0, time;
};

#endif
