/* Exercises compute_jvp (forward-mode Jacobian-vector product) and the adjoint
 * dot-product oracle on a GROWING two-parameter System, compiled on demand by
 * test-ad-jvp-oracle.R (sourceCpp).
 *
 * Toy: cohorts born at value b, each decaying at rate k; new cohorts introduced
 * mid-run (resize). Metric = sum of final states. Closed form (3 cohorts born at
 * t = 0,1,2, read at t = 3):
 *     M(k,b) = b (e^{-3k} + e^{-2k} + e^{-k})
 *     dM/dk  = b (-3 e^{-3k} - 2 e^{-2k} - e^{-k}),   dM/db = e^{-3k}+e^{-2k}+e^{-k}
 * The test checks: forward value == reverse value == analytic; the reverse
 * gradient == analytic; and the FD-free oracle identity  J v == <v, gradient>
 * (compute_jvp forward leg vs compute_gradient reverse leg) to machine precision.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/gradient.hpp>

#include <vector>
#include <memory>
#include <cmath>

using namespace Rcpp;

// Two params (decay k, birth value b); ode_size() grows as cohorts are introduced.
template <class S>
class GrowingToy {
public:
  using value_type = S;
  S k, b;
  std::vector<S> y;
  double time = 0.0, t0 = 0.0;

  GrowingToy(S k_, S b_) : k(k_), b(b_) { reset(); }

  std::size_t ode_size() const { return y.size(); }
  double ode_time() const { return time; }
  void reset() { y.assign(1, b); time = t0; }          // birth value b seeds the initial state
  void introduce() { y.push_back(b); }

  // The two low-level parameters the gradient is taken w.r.t. (§8.1 contract).
  std::vector<S*> ad_parameters()    { return {&k, &b}; }
  std::vector<S*> ad_initial_state() { return {}; }

  template <class It> It set_ode_state(It it, double t) {
    for (auto& yi : y) yi = *it++;
    time = t;
    return it;
  }
  template <class It> It ode_state(It it) const {
    for (auto const& yi : y) *it++ = yi;
    return it;
  }
  template <class It> It ode_rates(It it) const {
    for (auto const& yi : y) *it++ = -k * yi;
    return it;
  }
};

static std::vector<double> grid(double a, double b, int n) {
  std::vector<double> g;
  for (int i = 0; i <= n; ++i) g.push_back(a + (b - a) * i / n);
  return g;
}

// A runnable satisfying the compute_gradient / compute_jvp contract: get_system_ref,
// reset, run (+ a tape slot the reverse driver fills). run() replays the fixed
// 3-cohort schedule, introducing (resizing) between segments.
template <class S>
struct Runner {
  using value_type = S;
  odelia::ode::Solver<GrowingToy<S>> solver;
  std::unique_ptr<xad::Tape<double>> tape;

  explicit Runner(GrowingToy<S> sys) : solver(sys, odelia::ode::OdeControl()) {}

  GrowingToy<S>& get_system_ref()    { return solver.get_system_ref(); }
  std::vector<S*> ad_parameters()    { return solver.get_system_ref().ad_parameters(); }
  std::vector<S*> ad_initial_state() { return solver.get_system_ref().ad_initial_state(); }
  void reset() { solver.reset(); }
  void run() {
    auto& sys = solver.get_system_ref();
    solver.advance_fixed(grid(0.0, 1.0, 10));
    sys.introduce(); solver.set_state_from_system(); solver.advance_fixed(grid(1.0, 2.0, 10));
    sys.introduce(); solver.set_state_from_system(); solver.advance_fixed(grid(2.0, 3.0, 10));
  }
};

template <class S>
static S sum_final(GrowingToy<S>& sys) {
  std::vector<S> st(sys.ode_size());
  sys.ode_state(st.begin());
  S s(0.0);
  for (auto const& v : st) s += v;
  return s;
}

// [[Rcpp::export]]
Rcpp::List jvp_oracle_demo(double k = 0.3, double b = 1.0,
                           double vk = 0.37, double vb = 0.71) {
  using adS = xad::adj<double>::active_type;
  using fS  = xad::fwd<double>::active_type;

  odelia::ode::DifferentiationTargets targets;
  targets.params = {0, 1};          // k, b
  targets.values = {k, b};

  // Reverse: value + full gradient g = [dM/dk, dM/db].
  Runner<adS> r_rev{GrowingToy<adS>(adS(k), adS(b))};
  auto [value_rev, g] = odelia::ode::compute_gradient(
      r_rev, targets, [](Runner<adS>& r) -> adS { return sum_final(r.get_system_ref()); });

  // Forward: value + directional derivative J v along v = (vk, vb).
  Runner<fS> r_fwd{GrowingToy<fS>(fS(k), fS(b))};
  std::vector<double> v = {vk, vb};
  auto [value_fwd, jvp] = odelia::ode::compute_directional_derivative(
      r_fwd, targets, v, [](Runner<fS>& r) -> fS { return sum_final(r.get_system_ref()); });

  // Oracle: <J v, u> = <v, Jᵀ u>. For a scalar output (u = 1) this is  jvp == <v, g>.
  const double dot_vg = vk * g[0] + vb * g[1];

  // Analytic references.
  const double E  = std::exp(-3 * k) + std::exp(-2 * k) + std::exp(-k);
  const double gk = b * (-3 * std::exp(-3 * k) - 2 * std::exp(-2 * k) - std::exp(-k));
  const double gb = E;

  return Rcpp::List::create(
      Rcpp::Named("value_rev") = value_rev,
      Rcpp::Named("value_fwd") = value_fwd,
      Rcpp::Named("value_analytic") = b * E,
      Rcpp::Named("grad_k") = g[0], Rcpp::Named("grad_b") = g[1],
      Rcpp::Named("grad_k_analytic") = gk, Rcpp::Named("grad_b_analytic") = gb,
      Rcpp::Named("jvp") = jvp,
      Rcpp::Named("dot_v_grad") = dot_vg,
      Rcpp::Named("jvp_analytic") = vk * gk + vb * gb);
}
