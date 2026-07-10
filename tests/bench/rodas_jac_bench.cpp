// Benchmark: hand-rolled forward-mode Jacobian (ode_jacobian.hpp) vs the vendored
// xad::computeJacobian, and both against the rest of a RODAS step (LU factor + 6
// stage solves + RHS evals). Answers issue #35's open question: is the per-step
// Jacobian a hot spot worth a bespoke sweep, or is the XAD facility fine?
//
// Compiled on demand via Rcpp::sourceCpp (see run harness). Forward mode only.

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <functional>
#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_jacobian.hpp>
#include <odelia/ode_linalg.hpp>

using namespace odelia;

// A tunable-size stiff system: a stiff linear decay + linear coupling + a
// nonlinear cross term, so df/dy is dense and state-dependent (a real Jacobian,
// not a constant matrix).
template <typename T = double>
class StiffSys {
public:
  using value_type = T;
  StiffSys(int n_, std::vector<double> k_)
    : n(n_), k(std::move(k_)), y(n_, T(0)), t0(0.0), time(0.0) {}

  size_t ode_size() const { return static_cast<size_t>(n); }
  double ode_t0() const { return t0; }
  double ode_time() const { return time; }

  template <typename It>
  It set_ode_state(It it, double time_) {
    time = time_;
    for (int i = 0; i < n; ++i) y[i] = *it++;
    return it;
  }
  template <typename It>
  It set_initial_state(It it, double t0_ = 0.0) {
    t0 = t0_;
    for (int i = 0; i < n; ++i) y[i] = *it++;
    return it;
  }
  template <typename It>
  It ode_initial_state(It it) const { for (int i = 0; i < n; ++i) *it++ = y[i]; return it; }

  template <typename It>
  It ode_rates(It it) const {
    for (int i = 0; i < n; ++i) {
      const int j = (i + 1) % n;
      *it++ = T(-k[i]) * y[i] + T(0.1) * y[j] + y[i] * y[j];
    }
    return it;
  }

  std::vector<double> pars() const { return k; }

  template <class U> using rebind = StiffSys<U>;
  template <class U>
  rebind<U> rebind_from() const { return StiffSys<U>(n, k); }

private:
  int n;
  std::vector<double> k;
  std::vector<T> y;
  double t0, time;
};

// XAD-facility Jacobian: the swap ode_jacobian.hpp would make. Forward mode,
// tape-free, wrapping derivs in the std::function shape computeJacobian wants.
template <typename System>
static void jac_xad(const System& system, const std::vector<double>& y,
                    double t, std::vector<double>& J) {
  using tangent = typename xad::fwd<double>::active_type;
  auto twin = system.template rebind_from<tangent>();
  const size_t n = y.size();
  std::vector<tangent> in(n);
  for (size_t i = 0; i < n; ++i) in[i] = tangent(y[i]);
  std::function<std::vector<tangent>(std::vector<tangent>&)> foo =
      [&twin, t, n](std::vector<tangent>& v) {
        std::vector<tangent> d(n);
        ode::derivs(twin, v, d, t);
        return d;
      };
  auto jac = xad::computeJacobian(in, foo, n);
  J.assign(n * n, 0.0);
  for (size_t r = 0; r < n; ++r)
    for (size_t c = 0; c < n; ++c) J[r * n + c] = jac[r][c];
}

using clk = std::chrono::steady_clock;
static double sec_since(clk::time_point t0) {
  return std::chrono::duration<double>(clk::now() - t0).count();
}

// Run `fn` repeatedly for ~budget seconds; return mean microseconds per call.
template <typename F>
static double time_us(double budget, F&& fn) {
  // warm up
  for (int i = 0; i < 3; ++i) fn();
  long iters = 0;
  auto t0 = clk::now();
  do {
    for (int b = 0; b < 16; ++b) fn();
    iters += 16;
  } while (sec_since(t0) < budget);
  return sec_since(t0) / iters * 1e6;
}

// [[Rcpp::export]]
Rcpp::DataFrame rodas_jac_bench(std::vector<int> sizes, double budget = 0.15) {
  std::vector<int> out_n;
  std::vector<double> out_hand, out_xad, out_ratio, out_rest, out_hand_frac,
      out_xad_frac, out_maxdiff;

  for (int n : sizes) {
    // Stiff spectrum: decay rates spread over several decades.
    std::vector<double> k(n);
    for (int i = 0; i < n; ++i)
      k[i] = std::pow(10.0, -1.0 + 5.0 * (n > 1 ? double(i) / (n - 1) : 0.0));
    StiffSys<double> sys(n, k);

    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 0.5 + 0.3 * std::sin(0.7 * i);
    const double t = 0.3;

    ode::Jacobian<StiffSys<double>> jac;
    jac.resize(n);
    std::vector<double> Jh(n * n), Jx(n * n);
    jac.compute(sys, y, t, Jh);
    jac_xad(sys, y, t, Jx);
    double maxdiff = 0.0;
    for (int i = 0; i < n * n; ++i)
      maxdiff = std::max(maxdiff, std::abs(Jh[i] - Jx[i]));

    const double hand_us = time_us(budget, [&] { jac.compute(sys, y, t, Jh); });
    const double xad_us = time_us(budget, [&] { jac_xad(sys, y, t, Jx); });

    // Rest of a RODAS step: build W = fac*I - J, factor once, 6 stage solves,
    // and the ~6 RHS evals the stages need. This is what the Jacobian competes
    // with for step time.
    const double rest_us = time_us(budget, [&] {
      std::vector<double> W(Jh);
      const double fac = 1.0 / (0.01 * 0.25);
      for (int i = 0; i < n * n; ++i) W[i] = -W[i];
      for (int d = 0; d < n; ++d) W[d * n + d] += fac;
      std::vector<size_t> piv;
      ode::linalg::lu_decompose(W, n, piv);
      std::vector<double> rhs(y), x, dtmp(n);
      for (int s = 0; s < 6; ++s) ode::linalg::lu_solve(W, n, piv, rhs, x);
      for (int s = 0; s < 6; ++s) ode::derivs(sys, y, dtmp, t);
    });

    out_n.push_back(n);
    out_hand.push_back(hand_us);
    out_xad.push_back(xad_us);
    out_ratio.push_back(xad_us / hand_us);
    out_rest.push_back(rest_us);
    out_hand_frac.push_back(hand_us / (hand_us + rest_us));
    out_xad_frac.push_back(xad_us / (xad_us + rest_us));
    out_maxdiff.push_back(maxdiff);
  }

  return Rcpp::DataFrame::create(
      Rcpp::Named("n") = out_n,
      Rcpp::Named("hand_us") = out_hand,
      Rcpp::Named("xad_us") = out_xad,
      Rcpp::Named("xad_over_hand") = out_ratio,
      Rcpp::Named("rest_us") = out_rest,
      Rcpp::Named("hand_frac_step") = out_hand_frac,
      Rcpp::Named("xad_frac_step") = out_xad_frac,
      Rcpp::Named("jac_maxdiff") = out_maxdiff);
}
