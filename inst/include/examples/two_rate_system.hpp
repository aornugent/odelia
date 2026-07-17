#ifndef TWO_RATE_SYSTEM_HPP_
#define TWO_RATE_SYSTEM_HPP_

#include <odelia/mri.hpp>
#include <vector>
#include <cmath>

// A demonstrator with a genuine timescale split: two fast reservoirs that relax
// (rate k) toward the mean of M slow modes, and M slow modes that relax slowly
// (rates omega << k) toward the fast output. The coupling is dissipative, so the
// trajectory is a bounded relaxation to consensus -- accuracy differences are
// meaningful, not swamped by growth. The slow block scales with M while the fast
// block stays size 2, so a multirate step's cost barely grows with M where a
// single-rate step's grows linearly. It is a valid plain System too (rates
// compose the two blocks), so the existing adaptive solver gives a single-rate
// reference to check the macro step against. Templated on T so the same code runs
// at double or an active AD type.
template <typename T = double>
class TwoRateSystem {
public:
  using value_type = T;
  using vec = std::vector<T>;

  TwoRateSystem(double k_, int n_slow_)
    : k(k_), n_slow(n_slow_), omega(n_slow_),
      u(2), x(n_slow_), t0(0.0), time(0.0) {
    for (int j = 0; j < n_slow; ++j)
      omega[j] = 0.02 + 0.18 * (n_slow > 1 ? double(j) / (n_slow - 1) : 0.0);
    reset();
  }

  size_t fast_size() const { return 2; }
  size_t slow_size() const { return (size_t)n_slow; }
  size_t coupling_size() const { return 1; }
  size_t ode_size() const { return fast_size() + slow_size(); }

  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  // The slow signal the fast block reads: the mean of the slow modes. Linear and
  // homogeneous, so the macro step reuses it on the slow tendency.
  void aggregate(const vec& xs, vec& g) const {
    T s = T(0.0);
    for (const auto& xj : xs) s += xj;
    g[0] = s / double(n_slow);
  }

  void fast_rates(const vec& us, const vec& g, vec& du) const {
    du[0] = k * (g[0] - us[0]);
    du[1] = k * (us[0] - us[1]);
  }

  void slow_rates(const vec& xs, const vec& us, vec& dx) const {
    for (int j = 0; j < n_slow; ++j) dx[j] = omega[j] * (us[1] - xs[j]);
  }

  // Plain ODE interface: state laid out [u; x], rates compose the two blocks.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    time = time_;
    for (auto& ui : u) ui = *it++;
    for (auto& xj : x) xj = *it++;
    return it;
  }

  template <typename Iterator>
  Iterator ode_state(Iterator it) const {
    for (const auto& ui : u) *it++ = ui;
    for (const auto& xj : x) *it++ = xj;
    return it;
  }

  template <typename Iterator>
  Iterator ode_rates(Iterator it) const {
    vec g(1), du(2), dx(n_slow);
    aggregate(x, g);
    fast_rates(u, g, du);
    slow_rates(x, u, dx);
    for (const auto& d : du) *it++ = d;
    for (const auto& d : dx) *it++ = d;
    return it;
  }

  void reset() {
    for (auto& ui : u) ui = T(0.0);
    for (int j = 0; j < n_slow; ++j) x[j] = T(1.0 + 0.5 * std::cos(3.14159265358979 * j / n_slow));
    time = t0;
  }

  std::vector<double> pars() const { return {k, double(n_slow)}; }

private:
  double k;
  int n_slow;
  std::vector<double> omega;
  vec u, x;
  double t0, time;
};

#endif
