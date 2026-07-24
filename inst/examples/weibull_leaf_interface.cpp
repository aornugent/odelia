/* A self-contained miniature of the plant/TF24 leaf, built from odelia
 * primitives only (no plant dependency), compiled on demand by
 * test-example-weibull-leaf.R (sourceCpp).
 *
 * It reproduces the structure that made TF24 hard, and nothing else:
 *
 *   - Weibull hydraulics via odelia::incomplete_gamma. The soil->collar water
 *     supply is E_up(p) = kmax * (G(p) - G(psi_soil)), where G is the Weibull
 *     antiderivative G(m) = (b/c) gamma(1/c, (m/b)^c). Differentiating a shape
 *     trait c flows through incomplete_gamma's d/da (series) channel, which no
 *     tape library supplies.
 *   - A nested inner solve: the internal CO2 ci is defined by supply = demand
 *     A(ci) = gc*(ca - ci), solved off the tape and returned by
 *     odelia::implicit_value (the stomatal ci node). Its denominator
 *     A'(ci) + gc > 0 is strictly positive -- a well-posed root, not a fold.
 *   - An outer optimum: the collar tension p_star = argmax W(p) with
 *     W = assim(ci(p)) - hydraulic_cost(p). p_star is found off the tape and
 *     returned by odelia::implicit_value with the stationarity residual
 *     F(p) = dW/dp (a central difference of the assembled profit, exactly as
 *     the real interior node builds it); d(p_star)/dtheta = -F_theta/F_p (IFT).
 *
 * The certificate checks reverse-mode AD against a clean, re-optimising finite
 * difference (perturb a trait, re-solve p_star and ci, central difference) for
 * four trait channels, and separates the two envelope facts:
 *   - dW/dtheta at p_star is the DIRECT partial only (the d(p_star) term
 *     vanishes because dW/dp is zero at the optimum) -- the envelope theorem;
 *   - d(E_up)/dtheta needs the d(p_star)/dtheta term (soil uptake is NOT
 *     stationary in p_star) -- the envelope asymmetry the SCM soil feedback
 *     depends on.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/incomplete_gamma.hpp>
#include <odelia/implicit_node.hpp>

#include <algorithm>
#include <cmath>

using namespace Rcpp;
using odelia::util::to_passive;

// ---- fixed leaf constants (not differentiated) ----------------------------
namespace {
constexpr double B_WEIBULL = 2.0;  // Weibull scale (MPa)
constexpr double CA = 30.0;        // ambient CO2
constexpr double KM = 2.0;         // Michaelis half-saturation for assimilation
constexpr double ALPHA = 0.5;      // gc = alpha * E_up
constexpr double G1 = 2.0;         // hydraulic-cost scale
}  // namespace

// The four differentiated traits, carried as one scalar bundle so the same code
// serves double (references) and the active reverse type (AD).
template <class S>
struct Traits {
  S kmax, c, psi_soil, vcmax;
};

// Weibull antiderivative G(m) = (b/c) gamma(1/c, (m/b)^c); the running integral
// of the conductivity exp(-(m/b)^c). Pure odelia::incomplete_gamma.
template <class S>
static S weibull_G(const S& m, const S& c) {
  using std::exp;
  using std::log;
  if (to_passive(m) <= 0.0) return S(0.0);
  S Xc = exp(c * log(m / B_WEIBULL));  // (m/b)^c
  return (B_WEIBULL / c) * odelia::incomplete_gamma<S>(S(1.0) / c, Xc);
}

// Soil->collar supply at collar tension magnitude p.
template <class S>
static S uptake(const S& p, const Traits<S>& t) {
  return t.kmax * (weibull_G<S>(p, t.c) - weibull_G<S>(t.psi_soil, t.c));
}

// Michaelis assimilation demand A(ci).
template <class S>
static S assim(const S& ci, const Traits<S>& t) {
  return t.vcmax * ci / (ci + KM);
}

// Hydraulic cost: fractional conductivity loss at collar tension p, scaled by g1.
template <class S>
static S hydraulic_cost(const S& p, const Traits<S>& t) {
  using std::exp;
  using std::pow;
  return G1 * (S(1.0) - exp(-pow(p / B_WEIBULL, t.c)));
}

// Solve the inner ci root in double: A(ci) = gc*(ca - ci), monotone => bisection.
static double solve_ci_double(double gc, const Traits<double>& t) {
  auto resid = [&](double ci) { return assim<double>(ci, t) - gc * (CA - ci); };
  double lo = 0.0, hi = CA;
  for (int i = 0; i < 200; ++i) {
    double mid = 0.5 * (lo + hi);
    if (resid(mid) > 0.0) hi = mid; else lo = mid;
  }
  return 0.5 * (lo + hi);
}

// Assembled profit W(p) at a given collar tension, with ci an implicit_value
// node so its trait-response flows without recording the root-find. The ci
// anchor is solved fresh at the passive p (matching the real leaf, where each p
// gets its own converged inner state).
template <class S>
static S profit(const S& p, const Traits<S>& t) {
  const double gc_d = to_passive(ALPHA * uptake<S>(p, t));
  const Traits<double> td{to_passive(t.kmax), to_passive(t.c),
                          to_passive(t.psi_soil), to_passive(t.vcmax)};
  const double ci_star = solve_ci_double(gc_d, td);
  S gc = ALPHA * uptake<S>(p, t);
  S ci = odelia::implicit_value<S>(ci_star, [&](S c_i) {
    return assim<S>(c_i, t) - gc * (CA - c_i);
  });
  return assim<S>(ci, t) - hydraulic_cost<S>(p, t);
}

// dW/dp in double (central difference; matches the node's stationarity residual).
static double dWdp_double(double p, const Traits<double>& t) {
  const double e = 1e-4 * (std::abs(p) + 1.0);
  return (profit<double>(p + e, t) - profit<double>(p - e, t)) / (2.0 * e);
}

// Locate the interior optimum p* by BISECTION on dW/dp = 0. The derivative
// crosses zero with nonzero slope (W'' != 0), so this pins p* to ~machine
// precision -- unlike a golden-section max of the flat-topped W, whose location
// is only good to sqrt(tol) and would dominate the FD reference error.
static double solve_pstar_double(const Traits<double>& t, double lo, double hi) {
  double a = lo, b = hi, fa = dWdp_double(a, t), fb = dWdp_double(b, t);
  if (fa * fb > 0.0) {  // no interior bracket -- return the better endpoint
    return (std::abs(fa) < std::abs(fb)) ? a : b;
  }
  for (int i = 0; i < 200; ++i) {
    double m = 0.5 * (a + b), fm = dWdp_double(m, t);
    if (fa * fm <= 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
    if (b - a < 1e-13) break;
  }
  return 0.5 * (a + b);
}

// p* as an implicit_value node: F(p) = dW/dp (central difference of the
// assembled profit), root at the optimum. dp*/dtheta = -F_theta/F_p by the IFT.
template <class S>
static S pstar_node(double p_star, const Traits<S>& t) {
  const double eps = 1e-3 * (std::abs(p_star) + 1.0);
  return odelia::implicit_value<S>(p_star, [&](S p) {
    return (profit<S>(p + eps, t) - profit<S>(p - eps, t)) / (2.0 * eps);
  });
}

using RevS = xad::adj<double>::active_type;

// Reverse-mode gradient of a scalar leaf output w.r.t. the four traits, at the
// optimum p*. `which` selects the output: 0 = p*, 1 = profit(p*), 2 = uptake(p*).
static std::vector<double> ad_grad(int which, const Traits<double>& t0,
                                   double p_star) {
  xad::adj<double>::tape_type tape;
  Traits<RevS> t{t0.kmax, t0.c, t0.psi_soil, t0.vcmax};
  tape.registerInput(t.kmax); tape.registerInput(t.c);
  tape.registerInput(t.psi_soil); tape.registerInput(t.vcmax);
  tape.newRecording();
  RevS ps = pstar_node<RevS>(p_star, t);
  RevS out = (which == 0) ? ps
           : (which == 1) ? profit<RevS>(ps, t)
                          : uptake<RevS>(ps, t);
  tape.registerOutput(out);
  xad::derivative(out) = 1.0;
  tape.computeAdjoints();
  return {xad::derivative(t.kmax), xad::derivative(t.c),
          xad::derivative(t.psi_soil), xad::derivative(t.vcmax)};
}

// Re-optimising finite-difference reference: perturb trait k, re-solve p*, read
// the output, central difference. This is the Gate-0 correctness anchor.
static std::vector<double> fd_grad(int which, Traits<double> t0, double lo,
                                   double hi) {
  auto out_at = [&](const Traits<double>& t) {
    double ps = solve_pstar_double(t, lo, hi);
    return (which == 0) ? ps : (which == 1) ? profit<double>(ps, t) : uptake<double>(ps, t);
  };
  double* fields[4] = {&t0.kmax, &t0.c, &t0.psi_soil, &t0.vcmax};
  std::vector<double> g(4);
  for (int k = 0; k < 4; ++k) {
    const double v0 = *fields[k];
    const double h = 1e-6 * (std::abs(v0) + 1.0);
    *fields[k] = v0 + h; const double fp = out_at(t0);
    *fields[k] = v0 - h; const double fm = out_at(t0);
    *fields[k] = v0;
    g[k] = (fp - fm) / (2 * h);
  }
  return g;
}

// ===========================================================================
// (A) The bound / fold regime -- the "one genuinely new concept" (design 4.3).
//
// Above a critical collar tension the stem loses hydraulic function; the leaf
// cannot operate past p_crit. p_crit solves the branch-death condition
//   C(p) = cond(p) - k_crit = 0,   cond(p) = exp(-(p/b)^c),
// with dC/dp < 0 (sign-definite, regular -- NOT the near-zero denominator of a
// naive optimum-of-a-fold). When the unconstrained interior optimum would
// exceed p_crit, the leaf is PINNED at the bound: p* = p_crit. Two things then
// differ from the interior case:
//   - dp*/dtheta flows through C (the bound moves with traits), NOT through the
//     stationarity IFT -- a different implicit_value residual, the design's
//     "regime-detected continuity/branch-death residual";
//   - dW/dp != 0 at the bound, so the envelope cancellation is GONE: the profit
//     gradient now carries dp*/dtheta too. The interior demo cannot witness this
//     (there dW/dp = 0 by construction).
// ===========================================================================
namespace {
constexpr double K_CRIT = 0.30;  // residual-conductivity threshold at the bound
}

// Residual stem conductivity at collar tension p (the Weibull survival curve).
template <class S>
static S conductivity(const S& p, const S& c) {
  using std::exp;
  using std::pow;
  return exp(-pow(p / B_WEIBULL, c));
}

// Solve the bound p_crit in double: cond(p) - k_crit = 0, monotone decreasing.
static double solve_pcrit_double(const Traits<double>& t) {
  auto C = [&](double p) { return conductivity<double>(p, t.c) - K_CRIT; };
  double lo = 1e-6, hi = 50.0;
  for (int i = 0; i < 200; ++i) {
    double m = 0.5 * (lo + hi);
    if (C(m) > 0.0) lo = m; else hi = m;
    if (hi - lo < 1e-13) break;
  }
  return 0.5 * (lo + hi);
}

// p_crit as an implicit_value node on the branch-death condition. Denominator
// dC/dp = -cond*(c/b)*(p/b)^(c-1) is strictly negative -- regular, unlike the
// dF/dci -> 0 of a naive fold optimum. dp_crit/dtheta = -C_theta/C_p by the IFT.
template <class S>
static S pcrit_node(double p_crit, const Traits<S>& t) {
  return odelia::implicit_value<S>(p_crit, [&](S p) {
    return conductivity<S>(p, t.c) - S(K_CRIT);
  });
}

// p* under the hydraulic bound: min(interior optimum, p_crit). Whichever binds
// selects the regime; the demo below is parameterised so the bound binds.
static double solve_pstar_bounded(const Traits<double>& t, double lo, double hi) {
  const double p_int = solve_pstar_double(t, lo, hi);
  const double p_crit = solve_pcrit_double(t);
  return std::min(p_int, p_crit);
}

// Reverse-mode gradient in the bound regime: p* IS p_crit, so the p* node is the
// branch-death node. which: 0 = p*, 1 = profit(p*), 2 = uptake(p*).
static std::vector<double> ad_grad_bound(int which, const Traits<double>& t0,
                                         double p_crit) {
  xad::adj<double>::tape_type tape;
  Traits<RevS> t{t0.kmax, t0.c, t0.psi_soil, t0.vcmax};
  tape.registerInput(t.kmax); tape.registerInput(t.c);
  tape.registerInput(t.psi_soil); tape.registerInput(t.vcmax);
  tape.newRecording();
  RevS ps = pcrit_node<RevS>(p_crit, t);
  RevS out = (which == 0) ? ps
           : (which == 1) ? profit<RevS>(ps, t)
                          : uptake<RevS>(ps, t);
  tape.registerOutput(out);
  xad::derivative(out) = 1.0;
  tape.computeAdjoints();
  return {xad::derivative(t.kmax), xad::derivative(t.c),
          xad::derivative(t.psi_soil), xad::derivative(t.vcmax)};
}

static std::vector<double> fd_grad_bound(int which, Traits<double> t0, double lo,
                                         double hi) {
  auto out_at = [&](const Traits<double>& t) {
    double ps = solve_pstar_bounded(t, lo, hi);
    return (which == 0) ? ps : (which == 1) ? profit<double>(ps, t) : uptake<double>(ps, t);
  };
  double* fields[4] = {&t0.kmax, &t0.c, &t0.psi_soil, &t0.vcmax};
  std::vector<double> g(4);
  for (int k = 0; k < 4; ++k) {
    const double v0 = *fields[k];
    const double h = 1e-6 * (std::abs(v0) + 1.0);
    *fields[k] = v0 + h; const double fp = out_at(t0);
    *fields[k] = v0 - h; const double fm = out_at(t0);
    *fields[k] = v0;
    g[k] = (fp - fm) / (2 * h);
  }
  return g;
}

// ===========================================================================
// (B) Two-layer soil feedback -- the spatial channel that carried the historic
// sign error (design 5, "the adjoint of a feedback loop is a feedback loop").
//
// Per-layer uptake E_i(p) = k_i (G(p) - G(psi_i)) for layers with psi_i < p;
// E_up = sum of active layers -> gc = alpha E_up. Each layer's sink depends on
// p*, and p* depends on every trait, so d(E_i)/dtheta carries dp*/dtheta with a
// layer-specific sign -- the exact structure where a transposed sign silently
// flips a gradient. The layer-crossing at p = psi_i is a Leibniz breakpoint; the
// operating point keeps both layers active (p > psi_i) so the FD reference is
// clean. Six differentiated fields: k0,k1,psi0,psi1,c,vcmax.
// ===========================================================================
template <class S>
struct SoilTraits {
  S k0, k1, psi0, psi1, c, vcmax;
};

// One layer's uptake; contributes only when the collar out-tensions the soil.
template <class S>
static S uptake_layer(const S& p, const S& k_i, const S& psi_i, const S& c) {
  if (to_passive(p) <= to_passive(psi_i)) return S(0.0);  // inactive (breakpoint)
  return k_i * (weibull_G<S>(p, c) - weibull_G<S>(psi_i, c));
}

template <class S>
static S uptake_soil(const S& p, const SoilTraits<S>& t) {
  return uptake_layer<S>(p, t.k0, t.psi0, t.c) +
         uptake_layer<S>(p, t.k1, t.psi1, t.c);
}

static double solve_ci_soil_double(double gc, const SoilTraits<double>& t) {
  const Traits<double> tc{t.k0, t.c, t.psi0, t.vcmax};  // assim uses c/vcmax only
  return solve_ci_double(gc, tc);
}

template <class S>
static S profit_soil(const S& p, const SoilTraits<S>& t) {
  const double gc_d = to_passive(ALPHA * uptake_soil<S>(p, t));
  const SoilTraits<double> td{to_passive(t.k0), to_passive(t.k1),
                              to_passive(t.psi0), to_passive(t.psi1),
                              to_passive(t.c), to_passive(t.vcmax)};
  const double ci_star = solve_ci_soil_double(gc_d, td);
  const Traits<S> tc{t.k0, t.c, t.psi0, t.vcmax};  // assim(ci) reads c/vcmax
  S gc = ALPHA * uptake_soil<S>(p, t);
  S ci = odelia::implicit_value<S>(ci_star, [&](S c_i) {
    return assim<S>(c_i, tc) - gc * (CA - c_i);
  });
  return assim<S>(ci, tc) - hydraulic_cost<S>(p, tc);
}

static double dWdp_soil_double(double p, const SoilTraits<double>& t) {
  const double e = 1e-4 * (std::abs(p) + 1.0);
  return (profit_soil<double>(p + e, t) - profit_soil<double>(p - e, t)) / (2.0 * e);
}

static double solve_pstar_soil_double(const SoilTraits<double>& t, double lo, double hi) {
  double a = lo, b = hi, fa = dWdp_soil_double(a, t), fb = dWdp_soil_double(b, t);
  if (fa * fb > 0.0) return (std::abs(fa) < std::abs(fb)) ? a : b;
  for (int i = 0; i < 200; ++i) {
    double m = 0.5 * (a + b), fm = dWdp_soil_double(m, t);
    if (fa * fm <= 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
    if (b - a < 1e-13) break;
  }
  return 0.5 * (a + b);
}

template <class S>
static S pstar_soil_node(double p_star, const SoilTraits<S>& t) {
  const double eps = 1e-3 * (std::abs(p_star) + 1.0);
  return odelia::implicit_value<S>(p_star, [&](S p) {
    return (profit_soil<S>(p + eps, t) - profit_soil<S>(p - eps, t)) / (2.0 * eps);
  });
}

// which: 0 = p*, 1 = profit, 2 = E_up, 3 = E_layer0, 4 = E_layer1.
static std::vector<double> ad_grad_soil(int which, const SoilTraits<double>& t0,
                                        double p_star) {
  xad::adj<double>::tape_type tape;
  SoilTraits<RevS> t{t0.k0, t0.k1, t0.psi0, t0.psi1, t0.c, t0.vcmax};
  tape.registerInput(t.k0); tape.registerInput(t.k1);
  tape.registerInput(t.psi0); tape.registerInput(t.psi1);
  tape.registerInput(t.c); tape.registerInput(t.vcmax);
  tape.newRecording();
  RevS ps = pstar_soil_node<RevS>(p_star, t);
  RevS out = (which == 0) ? ps
           : (which == 1) ? profit_soil<RevS>(ps, t)
           : (which == 2) ? uptake_soil<RevS>(ps, t)
           : (which == 3) ? uptake_layer<RevS>(ps, t.k0, t.psi0, t.c)
                          : uptake_layer<RevS>(ps, t.k1, t.psi1, t.c);
  tape.registerOutput(out);
  xad::derivative(out) = 1.0;
  tape.computeAdjoints();
  return {xad::derivative(t.k0), xad::derivative(t.k1),
          xad::derivative(t.psi0), xad::derivative(t.psi1),
          xad::derivative(t.c), xad::derivative(t.vcmax)};
}

static std::vector<double> fd_grad_soil(int which, SoilTraits<double> t0,
                                        double lo, double hi) {
  auto out_at = [&](const SoilTraits<double>& t) {
    double ps = solve_pstar_soil_double(t, lo, hi);
    return (which == 0) ? ps
         : (which == 1) ? profit_soil<double>(ps, t)
         : (which == 2) ? uptake_soil<double>(ps, t)
         : (which == 3) ? uptake_layer<double>(ps, t.k0, t.psi0, t.c)
                        : uptake_layer<double>(ps, t.k1, t.psi1, t.c);
  };
  double* fields[6] = {&t0.k0, &t0.k1, &t0.psi0, &t0.psi1, &t0.c, &t0.vcmax};
  std::vector<double> g(6);
  for (int k = 0; k < 6; ++k) {
    const double v0 = *fields[k];
    const double h = 1e-6 * (std::abs(v0) + 1.0);
    *fields[k] = v0 + h; const double fp = out_at(t0);
    *fields[k] = v0 - h; const double fm = out_at(t0);
    *fields[k] = v0;
    g[k] = (fp - fm) / (2 * h);
  }
  return g;
}

// [[Rcpp::export]]
Rcpp::List weibull_leaf_demo(double kmax = 2.0, double c = 3.0,
                             double psi_soil = 0.4, double vcmax = 5.0) {
  const Traits<double> t0{kmax, c, psi_soil, vcmax};
  const double lo = psi_soil + 1e-3, hi = 6.0;
  const double p_star = solve_pstar_double(t0, lo, hi);
  // Interior-regime check: dW/dp ~ 0 at p* (not clamped to a bound).
  const double e = 1e-4;
  const double dW = (profit<double>(p_star + e, t0) - profit<double>(p_star - e, t0)) / (2 * e);

  auto nv = [](const std::vector<double>& x) {
    return Rcpp::NumericVector(x.begin(), x.end());
  };

  // Trait order is fixed: kmax, c, psi_soil, vcmax. reld is computed in R.
  return Rcpp::List::create(
      Rcpp::Named("traits") = Rcpp::CharacterVector::create("kmax", "c", "psi_soil", "vcmax"),
      Rcpp::Named("p_star") = p_star,
      Rcpp::Named("dW_dp_at_pstar") = dW,
      Rcpp::Named("dpstar_ad") = nv(ad_grad(0, t0, p_star)),
      Rcpp::Named("dpstar_fd") = nv(fd_grad(0, t0, lo, hi)),
      Rcpp::Named("dprofit_ad") = nv(ad_grad(1, t0, p_star)),
      Rcpp::Named("dprofit_fd") = nv(fd_grad(1, t0, lo, hi)),
      Rcpp::Named("duptake_ad") = nv(ad_grad(2, t0, p_star)),
      Rcpp::Named("duptake_fd") = nv(fd_grad(2, t0, lo, hi)));
}

// The bound / fold regime: vcmax pushes the unconstrained optimum past the
// hydraulic bound, so p* pins at p_crit and the profit gradient carries dp*.
// [[Rcpp::export]]
Rcpp::List weibull_leaf_bound_demo(double kmax = 2.0, double c = 3.0,
                                   double psi_soil = 0.4, double vcmax = 40.0) {
  const Traits<double> t0{kmax, c, psi_soil, vcmax};
  const double lo = psi_soil + 1e-3, hi = 6.0;
  const double p_int = solve_pstar_double(t0, lo, hi);
  const double p_crit = solve_pcrit_double(t0);
  const double p_star = std::min(p_int, p_crit);
  const double e = 1e-4;
  const double dW = (profit<double>(p_star + e, t0) - profit<double>(p_star - e, t0)) / (2 * e);

  auto nv = [](const std::vector<double>& x) {
    return Rcpp::NumericVector(x.begin(), x.end());
  };
  return Rcpp::List::create(
      Rcpp::Named("traits") = Rcpp::CharacterVector::create("kmax", "c", "psi_soil", "vcmax"),
      Rcpp::Named("p_star") = p_star,
      Rcpp::Named("p_interior") = p_int,
      Rcpp::Named("p_crit") = p_crit,
      Rcpp::Named("bound_binds") = (p_crit < p_int),
      Rcpp::Named("dW_dp_at_pstar") = dW,
      Rcpp::Named("dpstar_ad") = nv(ad_grad_bound(0, t0, p_crit)),
      Rcpp::Named("dpstar_fd") = nv(fd_grad_bound(0, t0, lo, hi)),
      Rcpp::Named("dprofit_ad") = nv(ad_grad_bound(1, t0, p_crit)),
      Rcpp::Named("dprofit_fd") = nv(fd_grad_bound(1, t0, lo, hi)),
      Rcpp::Named("duptake_ad") = nv(ad_grad_bound(2, t0, p_crit)),
      Rcpp::Named("duptake_fd") = nv(fd_grad_bound(2, t0, lo, hi)));
}

// Two-layer soil feedback: per-layer uptake, both layers active, the spatial
// (psi0/psi1) channel and per-layer sink gradients checked against FD.
// [[Rcpp::export]]
Rcpp::List weibull_leaf_soil_demo(double k0 = 1.2, double k1 = 0.8,
                                  double psi0 = 0.3, double psi1 = 0.9,
                                  double c = 3.0, double vcmax = 5.0) {
  const SoilTraits<double> t0{k0, k1, psi0, psi1, c, vcmax};
  const double lo = std::max(psi0, psi1) + 1e-3, hi = 6.0;
  const double p_star = solve_pstar_soil_double(t0, lo, hi);
  const double e = 1e-4;
  const double dW = (profit_soil<double>(p_star + e, t0) - profit_soil<double>(p_star - e, t0)) / (2 * e);

  auto nv = [](const std::vector<double>& x) {
    return Rcpp::NumericVector(x.begin(), x.end());
  };
  // Field order: k0, k1, psi0, psi1, c, vcmax.
  return Rcpp::List::create(
      Rcpp::Named("traits") = Rcpp::CharacterVector::create("k0", "k1", "psi0", "psi1", "c", "vcmax"),
      Rcpp::Named("p_star") = p_star,
      Rcpp::Named("dW_dp_at_pstar") = dW,
      Rcpp::Named("both_active") = (p_star > psi0 && p_star > psi1),
      Rcpp::Named("dpstar_ad") = nv(ad_grad_soil(0, t0, p_star)),
      Rcpp::Named("dpstar_fd") = nv(fd_grad_soil(0, t0, lo, hi)),
      Rcpp::Named("dprofit_ad") = nv(ad_grad_soil(1, t0, p_star)),
      Rcpp::Named("dprofit_fd") = nv(fd_grad_soil(1, t0, lo, hi)),
      Rcpp::Named("dEup_ad") = nv(ad_grad_soil(2, t0, p_star)),
      Rcpp::Named("dEup_fd") = nv(fd_grad_soil(2, t0, lo, hi)),
      Rcpp::Named("dE0_ad") = nv(ad_grad_soil(3, t0, p_star)),
      Rcpp::Named("dE0_fd") = nv(fd_grad_soil(3, t0, lo, hi)),
      Rcpp::Named("dE1_ad") = nv(ad_grad_soil(4, t0, p_star)),
      Rcpp::Named("dE1_fd") = nv(fd_grad_soil(4, t0, lo, hi)));
}
