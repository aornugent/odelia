/* An example exercising a GROWING-dimension System under reverse-mode AD,
 * compiled on demand by test-ad-growing-resize.R (sourceCpp).
 *
 * A toy System of cohorts, each decaying at rate k, with new cohorts introduced
 * mid-run (each introduction grows the ODE state vector via the solver's
 * resize). The reverse-mode gradient d(sum of final states)/dk must survive
 * those resizes and match the closed form and finite differences. It is checked
 * with reserve_state OFF (each introduction reallocates the state vector) and ON
 * (reserved to the final size, so resize never reallocates) -- confirming the
 * adjoint chain is preserved across a std::vector resize either way, and that
 * reserve_state is a memory optimisation, not a correctness requirement.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>

#include <vector>
#include <cmath>

using namespace Rcpp;

// Cohorts y_i decaying at a shared rate k: dy_i/dt = -k y_i. ode_size() grows as
// cohorts are introduced -- the growing-dimension System the SCM will be.
template <class S>
class ToyGrowingSystem {
public:
  using value_type = S;
  S k;
  std::vector<S> y;
  double time = 0.0, t0 = 0.0;

  explicit ToyGrowingSystem(S k_) : k(k_) { reset(); }

  std::size_t ode_size() const { return y.size(); }
  double ode_time() const { return time; }
  void reset() { y.assign(1, S(1.0)); time = t0; }
  void introduce() { y.push_back(S(1.0)); }

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

// 1 cohort on [0,1]; introduce -> 2 on [1,2]; introduce -> 3 on [2,3]. Returns
// the summed final state.
template <class S>
static S run_growing(odelia::ode::Solver<ToyGrowingSystem<S>>& solver, bool reserve) {
  if (reserve) solver.reserve_state(3);
  auto& sys = solver.get_system_ref();
  solver.reset();
  solver.advance_fixed(grid(0.0, 1.0, 10));
  sys.introduce(); solver.set_state_from_system();
  solver.advance_fixed(grid(1.0, 2.0, 10));
  sys.introduce(); solver.set_state_from_system();
  solver.advance_fixed(grid(2.0, 3.0, 10));
  std::vector<S> st(sys.ode_size());
  sys.ode_state(st.begin());
  S s(0.0);
  for (auto const& v : st) s += v;
  return s;
}

static double metric_double(double k, bool reserve) {
  ToyGrowingSystem<double> toy(k);
  odelia::ode::Solver<ToyGrowingSystem<double>> solver(toy, odelia::ode::OdeControl());
  return run_growing(solver, reserve);
}

static Rcpp::List one(double k, double delta, bool reserve) {
  using ad = xad::adj<double>;
  using S = ad::active_type;

  ToyGrowingSystem<S> toy(k);
  odelia::ode::Solver<ToyGrowingSystem<S>> solver(toy, odelia::ode::OdeControl());
  if (!solver.tape) solver.tape = std::make_unique<ad::tape_type>(false);
  solver.tape->activate();
  S kk = k;
  solver.tape->registerInput(kk);
  solver.tape->newRecording();
  solver.get_system_ref().k = kk;
  S m = run_growing(solver, reserve);
  solver.tape->registerOutput(m);
  xad::derivative(m) = 1.0;
  solver.tape->computeAdjoints();
  double ad_grad = xad::derivative(kk);
  solver.tape->deactivate();

  double fd = (metric_double(k + delta, reserve) -
               metric_double(k - delta, reserve)) / (2.0 * delta);
  double an = -(3 * std::exp(-3 * k) + 2 * std::exp(-2 * k) + std::exp(-k));

  return Rcpp::List::create(
      Rcpp::Named("value") = xad::value(m),
      Rcpp::Named("ad") = ad_grad,
      Rcpp::Named("fd") = fd,
      Rcpp::Named("analytic") = an);
}

// [[Rcpp::export]]
Rcpp::List growing_resize_demo(double k = 0.3, double delta = 1e-4) {
  return Rcpp::List::create(Rcpp::Named("no_reserve") = one(k, delta, false),
                            Rcpp::Named("reserve") = one(k, delta, true));
}
