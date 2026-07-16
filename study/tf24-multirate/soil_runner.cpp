// TF24 soil-water model driven by a daily rainfall sequence, integrated with
// odelia's explicit RKCK 4(5) and implicit RODAS 4(3) steppers.
//
// Physics is faithful to plant/inst/include/plant/models/tf24_environment.h:
//   K(theta)   = stiff * K_sat * (theta/theta_sat)^(2 n_psi + 3)   [drainage, mm/day]
//   psi(theta) = a_psi * (theta/theta_sat)^(-n_psi) / 1e6          [potential, MPa]
//   d theta_i/dt = (in_i - K(theta_i) - uptake_i) / dz             [per layer]
// with saturation-excess infiltration and a smooth root-uptake stress shutoff.
// Units are physically consistent: depths in mm (dz = 1500/5 = 300 mm), fluxes in
// mm/day, theta dimensionless, time in days.
//
// A block of M "reader" states integrates the aggregate matric potential,
// dx_k/dt = w_k * sum_i psi_i. psi diverges as theta -> residual, so these readers
// carry the near-singular feature into the error-controlled state (the accuracy
// amplifier), and M of them make the system size N = 5 + M (the N/L amplifier).

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>

using namespace odelia;

// RHS-evaluation counters, split by scalar type so the forward-AD Jacobian work
// (twin, FReal) is separable from the stepper's own stage evaluations (double).
static long g_rhs_double = 0;
static long g_rhs_twin   = 0;

// pow that resolves to std::pow for double and xad::pow for AD scalars (ADL).
// Base must be a concrete scalar (not an XAD expression); exponent is a double.
template <typename T>
static inline T powT(const T& a, double b) {
  using std::pow; using xad::pow;
  return pow(a, b);
}

// ---- TF24 parameters (header defaults) ------------------------------------
struct SoilPars {
  double theta_sat   = 0.428;
  double K_sat       = 163.0411;   // mm/day
  double a_psi       = 1.78e3;     // Pa
  double n_psi       = 6.57;
  double theta_res   = 1e-2;
  double depth_mm    = 1500.0;     // 1.5 m in mm (consistent with K in mm/day)
  double a_infil     = 1.0;
  double b_infil     = 8.0;
  double psi_max_MPa = 1e3;
  double theta_wilt  = 0.12;       // uptake stress shutoff floor
  double stiff       = 1.0;        // drainage-stiffness dial (x K_sat)
  double t_pot       = 4.0;        // potential transpiration demand, mm/day
};

template <typename T = double>
class SoilSystem {
public:
  using value_type = T;

  SoilSystem(SoilPars p_, size_t M_, const std::vector<double>& rain_)
    : p(p_), nL(5), M(M_), rain(rain_) {
    dz = p.depth_mm / double(nL);
    p_drain = 2.0 * p.n_psi + 3.0;   // 16.14
    p_ret   = -p.n_psi;              // -6.57
    // root fraction: shallow-weighted, normalised
    root = {0.35, 0.28, 0.20, 0.12, 0.05};
    // reader weights spread around 1 so the M states are distinct
    w.resize(M);
    for (size_t k = 0; k < M; ++k) w[k] = 0.5 + 1.0 * double(k) / std::max<size_t>(1, M);
    // initial state
    th_init.assign(nL, T(0.30 * p.theta_sat));   // start moderately dry
    x_init.assign(M, T(0.0));
    t0 = 0.0;
    reset();
  }

  size_t ode_size() const { return nL + M; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  // piecewise-constant daily rainfall (mm/day); blind to sub-day structure
  double rain_at(double t) const {
    long d = (long)std::floor(t);
    if (d < 0) d = 0;
    if (d >= (long)rain.size()) d = (long)rain.size() - 1;
    return rain[d];
  }

  T Kf(const T& th) const {
    T r = th / p.theta_sat;
    if (r < 0) r = T(0.0);
    return p.stiff * p.K_sat * powT(r, p_drain);
  }
  T psif(const T& th) const {
    T t = th; if (t < p.theta_res) t = T(p.theta_res);
    T ratio = t / p.theta_sat;               // concrete scalar base for powT
    T val = p.a_psi * powT(ratio, p_ret) / 1e6;
    if (val > p.psi_max_MPa) val = T(p.psi_max_MPa);
    return val;
  }
  // smooth uptake stress: 0 at residual, 1 at/above wilting (Hermite smoothstep)
  T stress(const T& th) const {
    T s = (th - p.theta_res) / (p.theta_wilt - p.theta_res);
    if (s <= 0) return T(0.0);
    if (s >= 1) return T(1.0);
    return s * s * (T(3.0) - T(2.0) * s);
  }

  void compute_rates() {
    if constexpr (std::is_same_v<T, double>) ++g_rhs_double; else ++g_rhs_twin;
    const double rain_t = rain_at(time);
    T r0 = th[0] / p.theta_sat; if (r0 < 0) r0 = T(0.0);
    T runoff = T(1.0) - p.a_infil * powT(r0, p.b_infil);
    if (runoff < 0) runoff = T(0.0);
    T infil = rain_t * runoff;

    // drainage out of each layer
    std::vector<T> out(nL);
    for (size_t i = 0; i < nL; ++i) out[i] = Kf(th[i]);

    T Psum = T(0.0);
    for (size_t i = 0; i < nL; ++i) Psum += psif(th[i]);

    for (size_t i = 0; i < nL; ++i) {
      T win = (i == 0) ? infil : out[i - 1];
      T uptake = p.t_pot * root[i] * stress(th[i]);
      dth[i] = (win - out[i] - uptake) / dz;
    }
    for (size_t k = 0; k < M; ++k) dx[k] = w[k] * Psum;
  }

  template <typename It>
  It set_ode_state(It it, double time_) {
    time = time_;
    for (size_t i = 0; i < nL; ++i) th[i] = *it++;
    for (size_t k = 0; k < M; ++k) x[k] = *it++;
    compute_rates();
    return it;
  }
  template <typename It>
  It set_initial_state(It it, double t0_ = 0.0) {
    t0 = t0_;
    for (size_t i = 0; i < nL; ++i) th_init[i] = *it++;
    for (size_t k = 0; k < M; ++k) x_init[k] = *it++;
    return it;
  }
  template <typename It> It ode_state(It it) const {
    for (size_t i = 0; i < nL; ++i) *it++ = th[i];
    for (size_t k = 0; k < M; ++k) *it++ = x[k];
    return it;
  }
  template <typename It> It ode_initial_state(It it) const {
    for (size_t i = 0; i < nL; ++i) *it++ = th_init[i];
    for (size_t k = 0; k < M; ++k) *it++ = x_init[k];
    return it;
  }
  template <typename It> It ode_rates(It it) const {
    for (size_t i = 0; i < nL; ++i) *it++ = dth[i];
    for (size_t k = 0; k < M; ++k) *it++ = dx[k];
    return it;
  }
  void reset() {
    th = th_init; x = x_init; dth.assign(nL, T(0.0)); dx.assign(M, T(0.0));
    time = t0; compute_rates();
  }
  std::vector<double> pars() const { return { p.stiff, p.t_pot }; }

  template <class S2> using rebind = SoilSystem<S2>;
  template <typename U> rebind<U> rebind_from() const {
    SoilSystem<U> s(p, M, rain);
    std::vector<U> init(nL + M);
    for (size_t i = 0; i < nL; ++i) init[i] = U(xad::value(th_init[i]));
    for (size_t k = 0; k < M; ++k) init[nL + k] = U(xad::value(x_init[k]));
    s.set_initial_state(init.begin(), t0);
    std::vector<U> st(nL + M);
    for (size_t i = 0; i < nL; ++i) st[i] = U(xad::value(th[i]));
    for (size_t k = 0; k < M; ++k) st[nL + k] = U(xad::value(x[k]));
    s.set_ode_state(st.begin(), time);
    return s;
  }

  size_t n_layers() const { return nL; }

private:
  SoilPars p;
  size_t nL, M;
  std::vector<double> rain;
  double dz, p_drain, p_ret, t0, time = 0.0;
  std::vector<double> root, w;
  std::vector<T> th, x, dth, dx, th_init, x_init;
};

static ode::Method pick(const std::string& m) {
  return (m == "rodas") ? ode::Method::rodas : ode::Method::rkck;
}

// Benchmark run: integrate over `grid` (adaptive between grid points; grid ends
// double as forcing-kink restarts). Returns effort + accuracy diagnostics and the
// daily theta trajectory (min over layers + layer 0).
// [[Rcpp::export]]
Rcpp::List soil_bench(std::vector<double> rain, std::vector<double> grid,
                      int M, double tol_rel, double tol_abs,
                      std::string method, double stiff, double t_pot) {
  SoilPars p; p.stiff = stiff; p.t_pot = t_pot;
  using Sys = SoilSystem<double>;
  Sys sys(p, (size_t)M, rain);

  ode::OdeControl ctrl(tol_abs, tol_rel, 1.0, 0.0, 1e-10, 1e6, 1e-4);
  ode::Solver<Sys> solver(sys, ctrl, pick(method));
  solver.set_collect(true);

  g_rhs_double = 0; g_rhs_twin = 0;
  auto t_start = std::chrono::high_resolution_clock::now();
  solver.advance_adaptive(grid);
  auto t_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  auto hist = solver.get_history();
  size_t nL = sys.n_layers();
  size_t ng = hist.size();
  Rcpp::NumericMatrix theta(ng, nL);
  Rcpp::NumericVector x0(ng), tgrid(ng);
  std::vector<double> s(nL + M);
  for (size_t i = 0; i < ng; ++i) {
    hist[i].ode_state(s.begin());
    for (size_t j = 0; j < nL; ++j) theta(i, j) = s[j];
    x0[i] = (M > 0) ? s[nL] : 0.0;
    tgrid[i] = grid[i];
  }

  return Rcpp::List::create(
    Rcpp::Named("method") = method,
    Rcpp::Named("M") = M,
    Rcpp::Named("stiff") = stiff,
    Rcpp::Named("n_steps") = (double)solver.times().size(),
    Rcpp::Named("wall_ms") = wall_ms,
    Rcpp::Named("rhs_double") = (double)g_rhs_double,
    Rcpp::Named("rhs_twin") = (double)g_rhs_twin,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("x0") = x0,
    Rcpp::Named("tgrid") = tgrid);
}

// Per-accepted-step log (single method), stepping day-by-day so forcing kinks sit
// on day boundaries and within-day steps reflect only stiffness/accuracy. Returns
// one row per accepted internal step: [t, h, theta_0..theta_{nL-1}].
// [[Rcpp::export]]
Rcpp::NumericMatrix soil_steplog(std::vector<double> rain, double T_end,
                                 int M, double tol_rel, double tol_abs,
                                 std::string method, double stiff, double t_pot) {
  SoilPars p; p.stiff = stiff; p.t_pot = t_pot;
  using Sys = SoilSystem<double>;
  Sys sys(p, (size_t)M, rain);
  ode::OdeControl ctrl(tol_abs, tol_rel, 1.0, 0.0, 1e-10, 1e6, 1e-4);
  ode::SolverInternal<Sys> si(sys, ctrl, pick(method));

  size_t nL = sys.n_layers();
  std::vector<double> rows;   // flattened, (2+nL) per row
  std::vector<double> st(nL + M);
  double t_prev = si.get_time();
  long nday = (long)std::floor(T_end);
  for (long d = 0; d < nday; ++d) {
    double day_end = (double)(d + 1);
    si.set_time_max(day_end);
    while (si.get_time() < day_end - 1e-12) {
      si.step(sys);
      double t = si.get_time();
      double h = t - t_prev; t_prev = t;
      st = si.get_state();
      rows.push_back(t); rows.push_back(h);
      for (size_t j = 0; j < nL; ++j) rows.push_back(st[j]);
    }
  }
  size_t ncol = 2 + nL;
  size_t nrow = rows.size() / ncol;
  Rcpp::NumericMatrix out(nrow, ncol);
  for (size_t i = 0; i < nrow; ++i)
    for (size_t j = 0; j < ncol; ++j) out(i, j) = rows[i * ncol + j];
  Rcpp::colnames(out) = Rcpp::CharacterVector::create("t","h","th0","th1","th2","th3","th4");
  return out;
}
