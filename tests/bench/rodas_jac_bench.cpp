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
#include <cstdlib>
#include <atomic>
#include <functional>
#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_jacobian.hpp>
#include <odelia/ode_linalg.hpp>

using namespace odelia;

// --- heap-allocation counter (process-global new/delete interposition) --------
// Diagnostic only: attributes the per-call overhead of each Jacobian path. We
// snapshot deltas around pure-C++ regions, so R's own allocations don't leak in.
static std::atomic<long long> g_allocs{0};
static std::atomic<long long> g_bytes{0};
void* operator new(std::size_t sz) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  g_bytes.fetch_add(static_cast<long long>(sz), std::memory_order_relaxed);
  void* p = std::malloc(sz ? sz : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t sz) { return ::operator new(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

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

  // Optional analytic Jacobian hook: df_i/dy_i = -k_i + y_{i+1},
  // df_i/dy_{i+1} = 0.1 + y_i. Exact, allocation-free, no AD machinery.
  void jacobian(const std::vector<double>& yv, double, std::vector<double>& J) const {
    J.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
      const int j = (i + 1) % n;
      J[i * n + i] += -k[i] + yv[j];
      J[i * n + j] += 0.1 + yv[i];
    }
  }

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

// Hand-rolled forward sweep on a twin built ONCE (the twin-caching fold): the
// same sweep as ode_jacobian.hpp minus the per-step rebind_from + scratch alloc.
template <typename Twin>
static void jac_hand_cached(Twin& twin, std::vector<typename Twin::value_type>& v,
                            std::vector<typename Twin::value_type>& d,
                            const std::vector<double>& y, double t,
                            std::vector<double>& J) {
  const size_t n = y.size();
  for (size_t j = 0; j < n; ++j) v[j] = typename Twin::value_type(y[j]);
  J.assign(n * n, 0.0);
  for (size_t col = 0; col < n; ++col) {
    xad::derivative(v[col]) = 1.0;
    ode::derivs(twin, v, d, t);
    for (size_t row = 0; row < n; ++row) J[row * n + col] = xad::derivative(d[row]);
    xad::derivative(v[col]) = 0.0;
  }
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
  std::vector<double> out_hand, out_xad, out_cached, out_analytic, out_rest,
      out_maxdiff;

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

    // Twin built once for the cached path and the analytic cross-check.
    using tan = typename xad::fwd<double>::active_type;
    auto twin = sys.template rebind_from<tan>();
    std::vector<tan> vv(n), dd(n);
    std::vector<double> Ja(n * n);
    sys.jacobian(y, t, Ja);
    double amaxdiff = 0.0;
    for (int i = 0; i < n * n; ++i)
      amaxdiff = std::max(amaxdiff, std::abs(Jh[i] - Ja[i]));
    maxdiff = std::max(maxdiff, amaxdiff);

    const double hand_us = time_us(budget, [&] { jac.compute(sys, y, t, Jh); });
    const double xad_us = time_us(budget, [&] { jac_xad(sys, y, t, Jx); });
    const double cached_us =
        time_us(budget, [&] { jac_hand_cached(twin, vv, dd, y, t, Jh); });
    const double analytic_us = time_us(budget, [&] { sys.jacobian(y, t, Ja); });

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
    out_cached.push_back(cached_us);
    out_analytic.push_back(analytic_us);
    out_rest.push_back(rest_us);
    out_maxdiff.push_back(maxdiff);
  }

  return Rcpp::DataFrame::create(
      Rcpp::Named("n") = out_n,
      Rcpp::Named("hand_us") = out_hand,
      Rcpp::Named("hand_cached_us") = out_cached,
      Rcpp::Named("xad_us") = out_xad,
      Rcpp::Named("analytic_us") = out_analytic,
      Rcpp::Named("rest_us") = out_rest,
      Rcpp::Named("jac_maxdiff") = out_maxdiff);
}

// Heap allocations per single Jacobian call (measured after warm-up).
// [[Rcpp::export]]
Rcpp::DataFrame rodas_jac_alloc(std::vector<int> sizes) {
  std::vector<int> out_n;
  std::vector<double> a_hand, a_cached, a_xad, a_analytic, b_xad;

  for (int n : sizes) {
    std::vector<double> k(n);
    for (int i = 0; i < n; ++i)
      k[i] = std::pow(10.0, -1.0 + 5.0 * (n > 1 ? double(i) / (n - 1) : 0.0));
    StiffSys<double> sys(n, k);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 0.5 + 0.3 * std::sin(0.7 * i);
    const double t = 0.3;

    ode::Jacobian<StiffSys<double>> jac;
    jac.resize(n);
    using tan = typename xad::fwd<double>::active_type;
    auto twin = sys.template rebind_from<tan>();
    std::vector<tan> vv(n), dd(n);
    std::vector<double> J(n * n);

    // Warm up every path so steady-state allocation is what we count.
    for (int w = 0; w < 5; ++w) {
      jac.compute(sys, y, t, J);
      jac_hand_cached(twin, vv, dd, y, t, J);
      jac_xad(sys, y, t, J);
      sys.jacobian(y, t, J);
    }

    auto count = [&](auto&& fn) {
      long long c0 = g_allocs.load();
      fn();
      return double(g_allocs.load() - c0);
    };
    auto bytes = [&](auto&& fn) {
      long long b0 = g_bytes.load();
      fn();
      return double(g_bytes.load() - b0);
    };

    out_n.push_back(n);
    a_hand.push_back(count([&] { jac.compute(sys, y, t, J); }));
    a_cached.push_back(count([&] { jac_hand_cached(twin, vv, dd, y, t, J); }));
    a_xad.push_back(count([&] { jac_xad(sys, y, t, J); }));
    a_analytic.push_back(count([&] { sys.jacobian(y, t, J); }));
    b_xad.push_back(bytes([&] { jac_xad(sys, y, t, J); }));
  }

  return Rcpp::DataFrame::create(
      Rcpp::Named("n") = out_n,
      Rcpp::Named("hand_allocs") = a_hand,
      Rcpp::Named("cached_allocs") = a_cached,
      Rcpp::Named("xad_allocs") = a_xad,
      Rcpp::Named("analytic_allocs") = a_analytic,
      Rcpp::Named("xad_bytes") = b_xad);
}
