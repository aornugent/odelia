/* Couples the toy Weibull leaf to a soil-water ODE, integrated through the odelia
 * Solver, compiled on demand by test-ad-soil-leaf.R (sourceCpp). This is the
 * plant-free witness for the design-live pieces the static leaf cannot exercise
 * (v3 §5, §3.2): the SOIL SUB-CYCLE ADJOINT over an integrated feedback loop, a
 * leaf node inside the rates, and the GROWING TAPE (a consumer introduced mid-run).
 *
 * The loop (the reason it cannot be tested without coupling): each soil layer i
 * holds water theta_i (an ODE state); its potential psi_soil,i = a (theta_sat/
 * theta_i)^n rises as the layer dries; the leaf reads psi_soil and draws per-layer
 * uptake E_i = kmax (G(p) - G(psi_soil,i)) through the SAME incomplete_gamma
 * hydraulics (so the shape trait c flows through the feedback); that uptake is the
 * soil sink dtheta_i/dt = (inflow - sum_plants E_i)/dz, which sets next step's
 * theta. "The adjoint of a feedback loop is a feedback loop." Each plant's biomass
 * grows on assimilation set by a ci implicit_value node (a leaf node living inside
 * ode_rates, recorded once per stage), and new plants are introduced mid-run so the
 * state (and tape) grows while the feedback is live.
 *
 * The certificate: reverse-mode d(final biomass)/d(kmax, c) matches a re-integrating
 * finite difference (perturb a trait, re-run the whole coupled ODE, central
 * difference) -- so the feedback adjoint, the node-in-rates, and the resize path are
 * all correct together. Tape memory is reported to show it grows linearly (bounded)
 * over the run, not explosively.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/incomplete_gamma.hpp>
#include <odelia/implicit_node.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/gradient.hpp>

#include <cmath>
#include <memory>
#include <vector>

using namespace Rcpp;
using odelia::util::to_passive;

namespace {
constexpr int L = 2;              // soil layers
constexpr double THETA_SAT = 1.0;
constexpr double PSI_A = 0.3;     // scale of the retention curve
constexpr double N_EXP = 1.5;     // retention exponent (stiffer as it dries)
constexpr double DZ = 1.0;        // layer depth
constexpr double INFLOW = 0.12;   // per-layer recharge
constexpr double B_WEIBULL = 2.0;
constexpr double P_COLLAR = 3.0;  // fixed collar tension (the leaf's draw head)
constexpr double CA = 30.0, KM = 2.0, ALPHA = 0.5;
constexpr double GROW = 0.02, MORT = 0.01;
constexpr double THETA0 = 0.6, W0 = 0.05;
}  // namespace

template <class S>
static S weibull_G(const S& m, const S& c) {
  using std::exp;
  using std::log;
  if (to_passive(m) <= 0.0) return S(0.0);
  S Xc = exp(c * log(m / B_WEIBULL));
  return (B_WEIBULL / c) * odelia::incomplete_gamma<S>(S(1.0) / c, Xc);
}

template <class S>
static S assim(const S& ci, double vcmax) { return vcmax * ci / (ci + KM); }

static double solve_ci_double(double gc, double vcmax) {
  auto resid = [&](double ci) { return assim<double>(ci, vcmax) - gc * (CA - ci); };
  double lo = 0.0, hi = CA;
  for (int i = 0; i < 200; ++i) {
    double mid = 0.5 * (lo + hi);
    if (resid(mid) > 0.0) hi = mid; else lo = mid;
  }
  return 0.5 * (lo + hi);
}

// The coupled soil + leaf System. Params: kmax (max conductance), c (Weibull shape).
// State layout: [theta_0 .. theta_{L-1}, w_0 .. w_{P-1}].
template <class S>
class SoilLeaf {
 public:
  using value_type = S;
  S kmax, cshape;
  std::vector<S> theta;  // soil water, L layers
  std::vector<S> w;      // plant biomass, grows via introduce()
  double time = 0.0, t0 = 0.0;
  double vcmax = 5.0;

  SoilLeaf(S k, S c) : kmax(k), cshape(c) { reset(); }

  std::size_t ode_size() const { return theta.size() + w.size(); }
  double ode_time() const { return time; }
  void reset() {
    theta.assign(L, S(THETA0));
    w.assign(1, S(W0));
    time = t0;
  }
  void introduce() { w.push_back(S(W0)); }

  std::vector<S*> ad_parameters() { return {&kmax, &cshape}; }
  std::vector<S*> ad_initial_state() { return {}; }

  template <class It>
  It set_ode_state(It it, double t) {
    for (auto& v : theta) v = *it++;
    for (auto& v : w) v = *it++;
    time = t;
    return it;
  }
  template <class It>
  It ode_state(It it) const {
    for (auto const& v : theta) *it++ = v;
    for (auto const& v : w) *it++ = v;
    return it;
  }

  // psi_soil for a layer from its water content (rises as theta falls).
  S psi_soil(std::size_t i) const {
    using std::pow;
    return S(PSI_A) * pow(S(THETA_SAT) / theta[i], S(N_EXP));
  }

  // One plant's uptake from one layer (0 if the layer out-tensions the collar).
  S uptake_layer(std::size_t i) const {
    S diff = weibull_G<S>(S(P_COLLAR), cshape) - weibull_G<S>(psi_soil(i), cshape);
    if (to_passive(diff) <= 0.0) return S(0.0);  // layer inactive (breakpoint)
    return kmax * diff;
  }

  template <class It>
  It ode_rates(It it) const {
    const std::size_t P = w.size();
    // Per-plant total uptake and the ci node that sets its assimilation.
    S E_total(0.0);
    std::vector<S> E_layer(L);
    for (std::size_t i = 0; i < L; ++i) { E_layer[i] = uptake_layer(i); E_total += E_layer[i]; }

    // gc from the transpiration stream; ci as an implicit_value node INSIDE the
    // rates (solved off-tape, recorded once here) -- the leaf node over the ODE.
    S gc = S(ALPHA) * E_total;
    const double ci_star = solve_ci_double(to_passive(gc), vcmax);
    const double vc = vcmax;
    S ci = odelia::implicit_value<S>(ci_star, [&](S c_i) {
      return assim<S>(c_i, vc) - gc * (CA - c_i);
    });
    S growth = S(GROW) * assim<S>(ci, vcmax);

    // Soil rates: each layer loses the sum over plants of its uptake (the sink that
    // closes the feedback). Plants are identical, so the layer sink is P * E_layer.
    for (std::size_t i = 0; i < L; ++i)
      *it++ = (S(INFLOW) - static_cast<double>(P) * E_layer[i]) / S(DZ);
    // Plant rates: grow on assimilation, decay on mortality.
    for (std::size_t j = 0; j < P; ++j)
      *it++ = growth - S(MORT) * w[j];
    return it;
  }
};

static std::vector<double> grid(double a, double b, int n) {
  std::vector<double> g;
  for (int i = 0; i <= n; ++i) g.push_back(a + (b - a) * i / n);
  return g;
}

template <class S>
struct Runner {
  using value_type = S;
  odelia::ode::Solver<SoilLeaf<S>> solver;
  std::unique_ptr<xad::Tape<double>> tape;

  explicit Runner(SoilLeaf<S> sys) : solver(sys, odelia::ode::OdeControl()) {}

  SoilLeaf<S>& get_system_ref() { return solver.get_system_ref(); }
  std::vector<S*> ad_parameters() { return solver.get_system_ref().ad_parameters(); }
  std::vector<S*> ad_initial_state() { return solver.get_system_ref().ad_initial_state(); }
  void reset() { solver.reset(); }
  void run() {
    auto& sys = solver.get_system_ref();
    solver.advance_fixed(grid(0.0, 1.0, 20));
    sys.introduce(); solver.set_state_from_system(); solver.advance_fixed(grid(1.0, 2.0, 20));
    sys.introduce(); solver.set_state_from_system(); solver.advance_fixed(grid(2.0, 3.0, 20));
  }
};

// Final total biomass -- the scalar functional the gradient is taken of.
template <class S>
static S total_biomass(SoilLeaf<S>& sys) {
  std::vector<S> st(sys.ode_size());
  sys.ode_state(st.begin());
  S s(0.0);
  for (std::size_t j = L; j < st.size(); ++j) s += st[j];  // skip the L soil states
  return s;
}

// Re-integrating finite difference: perturb a trait, re-run the whole coupled ODE.
static double run_biomass_double(double kmax, double c) {
  Runner<double> r{SoilLeaf<double>(kmax, c)};
  r.reset();
  r.run();
  return total_biomass(r.get_system_ref());
}

// [[Rcpp::export]]
Rcpp::List soil_leaf_demo(double kmax = 0.05, double c = 3.0) {
  using adS = xad::adj<double>::active_type;

  odelia::ode::DifferentiationTargets targets;
  targets.params = {0, 1};
  targets.values = {kmax, c};

  Runner<adS> r_rev{SoilLeaf<adS>(adS(kmax), adS(c))};
  auto [value_rev, g] = odelia::ode::compute_gradient(
      r_rev, targets, [](Runner<adS>& r) -> adS { return total_biomass(r.get_system_ref()); });
  const double tape_bytes = r_rev.tape ? static_cast<double>(r_rev.tape->getMemory()) : 0.0;

  // Re-integrating central FD across both traits.
  const double hk = 1e-6 * (std::abs(kmax) + 1.0);
  const double hc = 1e-6 * (std::abs(c) + 1.0);
  const double gk_fd = (run_biomass_double(kmax + hk, c) - run_biomass_double(kmax - hk, c)) / (2 * hk);
  const double gc_fd = (run_biomass_double(kmax, c + hc) - run_biomass_double(kmax, c - hc)) / (2 * hc);

  return Rcpp::List::create(
      Rcpp::Named("value") = value_rev,
      Rcpp::Named("value_double") = run_biomass_double(kmax, c),
      Rcpp::Named("grad_kmax") = g[0],
      Rcpp::Named("grad_c") = g[1],
      Rcpp::Named("grad_kmax_fd") = gk_fd,
      Rcpp::Named("grad_c_fd") = gc_fd,
      Rcpp::Named("tape_bytes") = tape_bytes);
}
