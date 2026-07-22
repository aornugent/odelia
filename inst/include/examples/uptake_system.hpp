#ifndef UPTAKE_SYSTEM_HPP_
#define UPTAKE_SYSTEM_HPP_

#include <odelia/mri.hpp>
#include <XAD/XAD.hpp>
#include <vector>
#include <cmath>

// A demonstrator shaped like the TF24 soil column under the T6 "uptake arbitrage".
// L fast layers u_l drain under a root-uptake sink a_l(u) -- a nonlinear, coupled
// function of the whole fast state (the toy stand-in for the O(M) cohort sum) --
// while gently refilling; M slow modes track the layer mean (the frozen-per-leg
// "cohort" block). The point the toy proves is NOT stiffness (the two-rate/drainage
// toys cover that) but the COUPLING REFRESH: over a macro leg the expensive a(u) is
// replaced by an affine model a0 + J*(u - anchor) captured once, and re-captured
// only when a trust monitor trips. The fast block therefore reads a *frozen*
// coupling (fast_rates_frozen), refreshed by refresh_anchor, and the monitor uses
// only cheap in-loop quantities (trust_excursion). true_a / trust_true_error expose
// the exact coupling so a full-resolve reference and an oracle re-expansion count
// can be measured against the cheap scheme. Templated on T so it runs at double or
// an active AD type; a_scale and the initial state are seedable inputs.
//
// Fast:  u_l' = -a_l(u)/dz_l + refill*(u_ref - u_l)
// Slow:  x_j' = w_j*(mean(u) - x_j)
// a_l(u) = a_scale * sum_k B[l][k] * u_k^2         (curvature -> the affine model
//                                                   drifts as u leaves the anchor)
// coupling_size()==0: the fast block does NOT read a linear aggregate of the slow
// block (as MRI-GARK does) -- it reads the frozen a(u), exactly as the plant patch
// does (assemble_resource_depletion, coupling_size()==0). freeze_slow is therefore
// unneeded: a depends on u only, and the slow block is frozen over the leg anyway.
template <typename T = double>
class UptakeSystem {
public:
  using value_type = T;
  using vec = std::vector<T>;

  UptakeSystem(double a_scale_, int n_fast_, int n_slow_)
    : a_scale(a_scale_), refill(0.05), u_ref(0.3), n_fast(n_fast_), n_slow(n_slow_),
      omega(n_slow_), dz(n_fast_), B(n_fast_ * n_fast_),
      u_init(n_fast_), x_init(n_slow_), u(n_fast_), x(n_slow_),
      a0_(n_fast_), J_(n_fast_ * n_fast_), anchor_(n_fast_),
      t0(0.0), time(0.0) {
    for (int j = 0; j < n_slow; ++j)
      omega[j] = 0.02 + 0.18 * (n_slow > 1 ? double(j) / (n_slow - 1) : 0.0);
    for (int l = 0; l < n_fast; ++l) dz[l] = 0.1 + 0.05 * l;   // deeper layers thicker
    // uptake coupling: dominant self term + weak neighbour draw, positive so a>0.
    for (int l = 0; l < n_fast; ++l)
      for (int k = 0; k < n_fast; ++k)
        B[l * n_fast + k] = (l == k) ? 0.5 : 0.05 / (1.0 + std::abs(l - k));
    for (int l = 0; l < n_fast; ++l) u_init[l] = T(0.9);   // wet: uptake active
    for (int j = 0; j < n_slow; ++j) x_init[j] = T(0.5 + 0.2 * std::cos(3.14159265358979 * j / n_slow));
    reset();
  }

  size_t fast_size() const { return (size_t)n_fast; }
  size_t slow_size() const { return (size_t)n_slow; }
  size_t coupling_size() const { return 0; }   // no linear aggregate; a reads u
  size_t ode_size() const { return fast_size() + slow_size(); }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  // Inert linear-aggregate channel (coupling_size()==0), present so the generic
  // macro stepper compiles; it carries nothing for this System.
  void aggregate(const vec&, vec&) const {}

  // The exact (expensive) coupling: the root-uptake sink per layer. On the real
  // patch this is the O(M) cohort sum; here a cheap nonlinear stand-in with real
  // curvature, so the affine refresh genuinely degrades away from the anchor.
  void true_a(const vec& us, vec& a) const {
    for (int l = 0; l < n_fast; ++l) {
      T s(0.0);
      for (int k = 0; k < n_fast; ++k) s += B[l * n_fast + k] * us[k] * us[k];
      a[l] = a_scale * s;
    }
  }
  // d(a_l)/d(u_k) = a_scale * B[l][k] * 2 u_k  (the byproduct Jacobian, Slice 3b-i).
  void true_a_jac(const vec& us, vec& Jout) const {
    for (int l = 0; l < n_fast; ++l)
      for (int k = 0; k < n_fast; ++k)
        Jout[l * n_fast + k] = a_scale * B[l * n_fast + k] * 2.0 * us[k];
  }

  // --- T6 uptake-inner hooks (used only by subcycle_uptake / UptakeSubcycle) ---
  // Capture the affine coupling model at the current fast state: one true_a +
  // one Jacobian (the single expensive evaluation the arbitrage rations).
  void refresh_anchor(const vec& us) {
    true_a(us, a0_);
    true_a_jac(us, J_);
    for (int l = 0; l < n_fast; ++l) anchor_[l] = us[l];
  }
  // The frozen coupling the fast block reads between re-expansions: a0 + J*(u-anchor).
  void predicted_a(const vec& us, vec& a) const {
    for (int l = 0; l < n_fast; ++l) {
      T s = a0_[l];
      for (int k = 0; k < n_fast; ++k) s += J_[l * n_fast + k] * (us[k] - anchor_[k]);
      a[l] = s;
    }
  }
  // Fast tendency reading the frozen (affine-refreshed) coupling.
  void fast_rates_frozen(const vec& us, vec& du) {
    vec a(n_fast);
    predicted_a(us, a);
    rate_from_a(us, a, du);
  }
  // Cheap, probe-free estimate of the affine model's RELATIVE a-error -- the number
  // the trust monitor compares against tol. The subtlety (the 3b-ii lesson): the
  // linearization error is SECOND order (~C*||du||^2, the curvature the linear model
  // misses), NOT the first-order excursion ||J*du|| (how much a *changes*). Using
  // the raw excursion over-triggers, because a genuinely moves a lot -- that is why
  // we track it. So estimate the 2nd-order remainder from the relative excursion
  // e = ||predicted_a - a0|| / ||a0|| (= ||J*du||/||a0||, sensitivity-scaled) and
  // return e^2: the leading error scales as e^2 up to an O(1) relative-curvature
  // factor, so one tol serves across regimes without any true_a probe. Conservative
  // by that O(1) factor (cheap count >= oracle), which is the safe direction.
  double trust_excursion(const vec& us) const {
    vec ap(n_fast);
    predicted_a(us, ap);
    double dan = 0.0, a0n = 0.0;
    for (int l = 0; l < n_fast; ++l) {
      dan = std::max(dan, std::abs(xad::value(ap[l]) - xad::value(a0_[l])));
      a0n = std::max(a0n, std::abs(xad::value(a0_[l])));
    }
    const double e = dan / std::max(a0n, 1e-30);
    return e * e;
  }
  // The oracle trust estimate (diagnostic only): the TRUE relative a-error the
  // affine model incurs at us. Costs a true_a evaluation, so it is never used in
  // production -- only to measure the oracle re-expansion count the cheap monitor
  // is graded against, and the accuracy the refresh achieves.
  double trust_true_error(const vec& us) const {
    vec at(n_fast), ap(n_fast);
    true_a(us, at);
    predicted_a(us, ap);
    double num = 0.0, den = 0.0;
    for (int l = 0; l < n_fast; ++l) {
      num = std::max(num, std::abs(xad::value(ap[l]) - xad::value(at[l])));
      den = std::max(den, std::abs(xad::value(at[l])));
    }
    return num / std::max(den, 1e-30);
  }

  // Trust threshold / micro-steps for the method="mri_uptake" Solver path. The toy
  // drives the uptake inner directly via mri_advance (see uptake_mri) so these are
  // only here to satisfy MriUptakeStep instantiation when a Solver is built over
  // this System (e.g. the adaptive reference); the direct path passes its own.
  double mri_uptake_tol() const { return 1e-2; }
  int mri_uptake_nmicro() const { return 40; }

  void slow_rates(const vec& xs, const vec& us, vec& dx) const {
    T m(0.0);
    for (const auto& ul : us) m += ul;
    m /= double(n_fast);
    for (int j = 0; j < n_slow; ++j) dx[j] = omega[j] * (m - xs[j]);
  }
  // Fast tendency reading the EXACT coupling (the full-resolve path: reference and
  // the plain ODE interface). This is the O(M)-every-step cost the arbitrage avoids.
  void fast_rates_true(const vec& us, vec& du) const {
    vec a(n_fast);
    true_a(us, a);
    rate_from_a(us, a, du);
  }
  // Generic multirate fast-rate hook (the linear-aggregate g is empty here,
  // coupling_size()==0): reads the exact coupling. Keeps the System usable by the
  // black-box AdaptiveSubcycle / method="mri" paths; the arbitrage uses the frozen
  // path (fast_rates_frozen via UptakeSubcycle) instead.
  void fast_rates(const vec& us, const vec& /*g*/, vec& du) const {
    fast_rates_true(us, du);
  }

  // Plain ODE interface (state [slow x | fast u]): the exact coupling, so the
  // adaptive solver gives a full-resolve reference to grade the macro scheme.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    time = time_;
    for (auto& xj : x) xj = *it++;
    for (auto& ul : u) ul = *it++;
    return it;
  }
  template <typename Iterator>
  Iterator ode_state(Iterator it) const {
    for (const auto& xj : x) *it++ = xj;
    for (const auto& ul : u) *it++ = ul;
    return it;
  }
  template <typename Iterator>
  Iterator ode_rates(Iterator it) const {
    vec du(n_fast), dx(n_slow);
    fast_rates_true(u, du);
    slow_rates(x, u, dx);
    for (const auto& d : dx) *it++ = d;
    for (const auto& d : du) *it++ = d;
    return it;
  }

  template <typename Iterator>
  Iterator set_params(Iterator it) { a_scale = *it++; return it; }
  template <typename Tape, typename Iterator>
  std::vector<T*> set_params(Tape& tape, Iterator it) {
    a_scale = *it++;
    tape.registerInput(a_scale);
    return {&a_scale};
  }

  template <typename Iterator>
  Iterator set_initial_state(Iterator it, double t0_ = 0.0) {
    t0 = t0_;
    for (auto& xj : x_init) xj = *it++;
    for (auto& ul : u_init) ul = *it++;
    return it;
  }
  template <typename Tape, typename Iterator>
  std::vector<T*> set_initial_state(Tape& tape, Iterator it, double t0_ = 0.0) {
    t0 = t0_;
    std::vector<T*> refs;
    for (auto& xj : x_init) { xj = *it++; tape.registerInput(xj); refs.push_back(&xj); }
    for (auto& ul : u_init) { ul = *it++; tape.registerInput(ul); refs.push_back(&ul); }
    return refs;
  }

  void reset() {
    for (int l = 0; l < n_fast; ++l) u[l] = u_init[l];
    for (int j = 0; j < n_slow; ++j) x[j] = x_init[j];
    time = t0;
  }

  std::vector<double> pars() const { return {xad::value(a_scale), double(n_fast), double(n_slow)}; }

private:
  // The fast tendency given a coupling a: uptake sink -a/dz plus a gentle refill.
  // The single place this physics lives -- the frozen and exact paths differ only
  // in how they obtain a, not in the tendency they build from it.
  void rate_from_a(const vec& us, const vec& a, vec& du) const {
    for (int l = 0; l < n_fast; ++l)
      du[l] = -a[l] / dz[l] + refill * (u_ref - us[l]);
  }

  T a_scale;
  double refill, u_ref;
  int n_fast, n_slow;
  std::vector<double> omega, dz, B;
  vec u_init, x_init, u, x;
  vec a0_, J_, anchor_;   // the captured affine coupling model (frozen per leg)
  double t0, time;
};

#endif
