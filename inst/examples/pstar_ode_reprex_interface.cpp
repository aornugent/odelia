/* Plant-free reprex for the TF24 interior-p* reverse-sweep failure.
 *
 * WHAT IT REPRODUCES
 * ------------------
 * plant's TF24 leaf assembly places an "interior optimum" node inside the ODE
 * rates: the collar tension p* is defined by the stationarity residual dW/dp = 0,
 * which `implicit_value` receives as a CENTRAL DIFFERENCE of a reduced profit --
 * and that reduced profit itself contains nested `implicit_value` nodes (the
 * psi_stem inversion and the ci root), each preceded by an off-tape double solve.
 * Replayed over an ODE schedule through `compute_gradient`, the reverse sweep read
 * a garbage operand slot and segfaulted. The cause is recorded below -- it was the
 * value-graft, not this composition, and this harness is what established that.
 *
 * The existing `weibull_leaf` toy nests implicit_value the same way but evaluates
 * it ONCE on a fresh tape, and passes -- so the single-shot composition is not
 * the trigger. This reprex adds the two things plant has and that toy lacks: the
 * node lives INSIDE `ode_rates` (re-recorded per stage over a schedule), and the
 * p* residual is a central difference that rebuilds the nested nodes twice per
 * evaluation. It exists so this failure can be bisected in seconds (sourceCpp)
 * instead of a ~6 minute plant rebuild.
 *
 * KNOBS (all exposed to R, so each suspected factor can be turned off alone)
 *   use_pstar : 1 = p* is an implicit_value node; 0 = p* is a frozen constant.
 *               In plant, 0 makes the crash disappear -- this is THE discriminator.
 *   nest      : 1 = the p* residual builds nested implicit_value nodes (psi_stem,
 *               ci); 0 = it uses the converged doubles instead (closed form only).
 *   n_steps   : ODE steps per stage -> tape size / number of re-recordings.
 *   nlayer    : soil layers -> expression depth per evaluation.
 *   persist   : 1 = a member vector<S> is read back from the PREVIOUS rates call
 *               before being overwritten (plant's soil_consumption_active_), so
 *               member actives carry slots across recordings.
 *   nfields   : count of persistent UNSEEDED member actives (INVALID_SLOT) mixed
 *               into the same expressions as seeded ones (plant's pars fields).
 *   graft     : 1 = replace each output value with a separately-known double,
 *               keeping only the derivative, via a lambda that DECLARES `-> S`
 *               (the safe form, and what odelia::util::graft_value does).
 *               2 = the same graft written with a DEDUCED return type. This is
 *               the root-cause bug, reproduced on demand: expect a segfault in
 *               the reverse sweep or a silently wrong gradient. See below.
 *
 * Returns the reverse gradient plus a re-integrating finite difference, so a run
 * that does NOT crash is still checked for correctness rather than just survival.
 *
 * ROOT CAUSE (found; reproduced here on demand with graft=2)
 * ---------------------------------------------------------
 * plant's value-graft was written as a lambda with a DEDUCED return type:
 *
 *     auto anchor = [](double v, S x) { return S(v) + (x - to_passive(x)); };
 *
 * XAD operators return expression templates holding references to their operands,
 * so this returns references to the temporary S(v) and to the by-value parameter
 * x -- both dead when the lambda returns. The caller materialises a dangling
 * expression and records whatever the reused stack now holds as an operand slot;
 * the reverse sweep dereferences it (derivatives_[~1e9]) and segfaults. The
 * dangling storage is STACK, so valgrind cannot see it, and the bogus slot varies
 * run to run. Declaring `-> S` fixes it; odelia::util::graft_value now owns the
 * idiom so it need not be hand-written, and AGENTS.md carries the rule.
 *
 * Why this took so long, and what the knobs below are still good for: because the
 * corruption depended on stack layout, switching almost anything off made the
 * crash "disappear" without being the cause. Freezing p*, dropping the nested
 * nodes, removing the soil uptake, and de-statefulling the double solvers each
 * looked like a fix. Every one of those was a false lead, and this harness is what
 * proved it -- each axis below runs clean with graft=1 (the safe graft), so none of
 * them was ever sufficient to cause the failure:
 *   - an interior-optimum implicit_value whose residual is a central difference;
 *   - nested implicit_value inside that residual, re-anchored by off-tape double
 *     solves at each perturbed p;
 *   - the node living inside ode_rates, re-recorded over a schedule replayed by
 *     compute_gradient (n_steps to 300);
 *   - expression depth via per-layer incomplete_gamma hydraulics (nlayer to 8);
 *   - persistent member actives reused across recordings (persist=1);
 *   - unseeded INVALID_SLOT member actives on the same expressions (nfields=40);
 *   - stateful vs pure double anchor solves (checked directly in plant).
 *
 * Keep this file as the regression witness for the bug class: graft=1 must stay
 * clean, and graft=2 documents (and, run under a debugger, demonstrates) what the
 * deduced return type does. The test file exercises graft=1 only, since graft=2
 * is undefined behaviour and would take the test process down with it.
 *
 * Note the expected accuracy signature: use_pstar=0 matches FD to ~1e-9, while
 * use_pstar=1 sits at ~7e-3 -- that is the eps=1e-2 central-difference error in
 * the stationarity residual, not a bug.
 */

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>

#include <odelia/incomplete_gamma.hpp>
#include <odelia/implicit_node.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/gradient.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>
#include <memory>
#include <vector>

using namespace Rcpp;
using odelia::util::to_passive;

namespace {

constexpr double THETA_SAT = 1.0;
constexpr double THETA0 = 0.8;
constexpr double W0 = 0.5;
constexpr double PSI_A = 0.3;
constexpr double N_EXP = 1.5;
constexpr double DZ = 1.0;
constexpr double INFLOW = 0.12;
constexpr double B_WEIB = 2.0;
constexpr double CA = 40.0;
constexpr double ALPHA = 0.02;
constexpr double BETA_STEM = 0.9;
constexpr double G1_COST = 1.5;
constexpr double BETA2 = 2.0;

// Cumulative Weibull vulnerability G(m) = int_0^m exp(-(s/b)^c) ds, the same
// incomplete_gamma channel plant's hydraulics use (carries the d/dc series).
template <class S>
S weibull_G(S m, S c) {
  using std::pow;
  if (to_passive(m) <= 0.0) return S(0.0);
  return (S(B_WEIB) / c) * odelia::incomplete_gamma<S>(S(1.0) / c, pow(m / S(B_WEIB), c));
}

template <class S>
S assim(S ci, S vcmax) {
  return vcmax * ci / (ci + S(15.0));
}

// Off-tape double solves, mirroring plant's find_psi_stem_from_psi_root /
// psi_stem_to_ci: the anchors the implicit_value nodes are grafted at.
double solve_ci_double(double gc, double vcmax) {
  double lo = 0.1, hi = CA;
  for (int i = 0; i < 80; ++i) {
    double m = 0.5 * (lo + hi);
    if (vcmax * m / (m + 15.0) - gc * (CA - m) < 0.0) lo = m; else hi = m;
  }
  return 0.5 * (lo + hi);
}

double solve_psistem_double(double demand, double c, double kmax) {
  double lo = 0.0, hi = 12.0;
  for (int i = 0; i < 80; ++i) {
    double m = 0.5 * (lo + hi);
    if (kmax * to_passive(weibull_G<double>(m, c)) < demand) lo = m; else hi = m;
  }
  return 0.5 * (lo + hi);
}

std::vector<double> grid(double a, double b, int n) {
  std::vector<double> t(n + 1);
  for (int i = 0; i <= n; ++i) t[i] = a + (b - a) * i / n;
  return t;
}

// --------------------------------------------------------------------------
// The System: soil water + biomass, whose rates contain the p* node.
// --------------------------------------------------------------------------
template <class S>
class PstarOde {
 public:
  using value_type = S;
  S kmax, cshape;
  std::vector<S> theta;
  std::vector<S> w;
  double time = 0.0, t0 = 0.0;
  double vcmax = 5.0;
  // knobs
  int use_pstar = 1, nest = 1, nlayer = 2, persist = 0, nfields = 0, graft = 0;

  // Persistent member actives, mirroring plant's TF24_Strategy_ members that
  // outlive a recording: soil_consumption_active_ (written by the leaf assembly,
  // read back by evapotranspiration_dt) and the unseeded pars fields (registered
  // as AD inputs but with INVALID_SLOT when not the seeded target).
  mutable std::vector<S> cons_cache;
  mutable std::vector<S> fields;

  PstarOde(S k, S c, int use_pstar_, int nest_, int nlayer_, int persist_, int nfields_,
           int graft_)
      : kmax(k), cshape(c), use_pstar(use_pstar_), nest(nest_), nlayer(nlayer_),
        persist(persist_), nfields(nfields_), graft(graft_) {
    fields.assign(nfields > 0 ? nfields : 0, S(1.0));
    reset();
  }

  std::size_t ode_size() const { return theta.size() + w.size(); }
  double ode_time() const { return time; }
  void reset() {
    theta.assign(nlayer, S(THETA0));
    w.assign(1, S(W0));
    time = t0;
  }

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

  S psi_soil(std::size_t i) const {
    using std::pow;
    return S(PSI_A) * pow(S(THETA_SAT) / theta[i], S(N_EXP));
  }

  // Per-layer uptake at a given collar tension; sums to the collar supply.
  S uptake(S p, std::vector<S>* per_layer) const {
    S total(0.0);
    for (int i = 0; i < nlayer; ++i) {
      S diff = weibull_G<S>(p, cshape) - weibull_G<S>(psi_soil(i), cshape);
      S e = (to_passive(diff) <= 0.0) ? S(0.0) : kmax * diff;
      if (per_layer) (*per_layer)[i] = e;
      total += e;
    }
    return total;
  }

  // The reduced profit at a collar tension p. With nest=1 this rebuilds the two
  // nested implicit_value nodes (psi_stem inversion, ci root) at doubles re-solved
  // off-tape per p -- exactly plant's profit_reduced.
  S profit_reduced(S p) const {
    S E = uptake(p, nullptr);
    S demand = S(BETA_STEM) * E;
    S psi_stem, ci;
    const double c_d = to_passive(cshape), k_d = to_passive(kmax);
    const double pss = solve_psistem_double(to_passive(demand), c_d, k_d);
    S gc = S(ALPHA) * E;
    const double ci_star = solve_ci_double(to_passive(gc), vcmax);
    if (nest) {
      const S kmax_l = kmax, c_l = cshape, demand_l = demand;
      psi_stem = odelia::implicit_value<S>(pss, [&](S ps) -> S {
        return kmax_l * weibull_G<S>(ps, c_l) - demand_l;
      });
      const S gc_l = gc;
      const double vc = vcmax;
      ci = odelia::implicit_value<S>(ci_star, [&](S c_i) -> S {
        return assim<S>(c_i, S(vc)) - gc_l * (S(CA) - c_i);
      });
    } else {
      psi_stem = S(pss);
      ci = S(ci_star);
    }
    using std::pow;
    S cost = S(G1_COST) * pow(S(1.0) - exp(-pow(psi_stem / S(B_WEIB), cshape)), S(BETA2));
    return assim<S>(ci, S(vcmax)) - cost;
  }

  // p*: the interior optimum, as an implicit_value on the stationarity residual
  // dW/dp expressed as a central difference (plant's exact construction).
  S pstar() const {
    const double p_anchor = 3.0;
    if (!use_pstar) return S(p_anchor);
    const double eps = 1e-2 * (std::abs(p_anchor) + 1.0);
    return odelia::implicit_value<S>(p_anchor, [&](S p) -> S {
      return (profit_reduced(p + eps) - profit_reduced(p - eps)) / S(2.0 * eps);
    });
  }

  template <class It>
  It ode_rates(It it) const {
    S p = pstar();
    std::vector<S> per_layer(nlayer);
    S E = uptake(p, &per_layer);
    S profit = profit_reduced(p);

    // Value-graft: replace the assembled value with a separately-known double,
    // keeping only the derivative -- plant's `anchor(leaf.profit_, assembled)`.
    // Two AD leaves per grafted output, matching the 2-operand statement plant
    // crashes on.
    if (graft == 1) {
      // SAFE: `-> S` materialises the graft while its operands are alive.
      auto anchor = [](double v, const S& x) -> S { return odelia::util::graft_value<S>(v, x); };
      profit = anchor(to_passive(profit), profit);
      for (int i = 0; i < nlayer; ++i) per_layer[i] = anchor(to_passive(per_layer[i]), per_layer[i]);
    } else if (graft == 2) {
      // UNSAFE, ON PURPOSE -- this is the bug this reprex exists to pin down.
      // No declared return type, so the lambda hands back an expression template
      // referencing the temporary S(v) and the by-value parameter x, both dead by
      // the time the caller materialises it. Expect a garbage operand slot and a
      // segfault in the reverse sweep (or, being undefined behaviour, a silently
      // wrong gradient). Only reachable by explicitly asking for graft=2.
      auto anchor_bad = [](double v, S x) { return S(v) + (x - to_passive(x)); };
      profit = anchor_bad(to_passive(profit), profit);
      for (int i = 0; i < nlayer; ++i)
        per_layer[i] = anchor_bad(to_passive(per_layer[i]), per_layer[i]);
    }

    // Mix in the unseeded member fields (INVALID_SLOT actives on the same
    // expressions as seeded ones), as plant's pars fields are.
    for (int f = 0; f < nfields; ++f) profit = profit + fields[f] * S(1e-12);

    // Read back LAST call's cached per-layer uptake before overwriting it: the
    // member actives then carry slots from an earlier recording, exactly as
    // plant's soil_consumption_active_ does across compute_jacobian's replays.
    if (persist) {
      if (cons_cache.size() == static_cast<std::size_t>(nlayer))
        for (int i = 0; i < nlayer; ++i) per_layer[i] = per_layer[i] + cons_cache[i] * S(1e-12);
      cons_cache.assign(per_layer.begin(), per_layer.end());
    }

    for (int i = 0; i < nlayer; ++i)
      *it++ = (S(INFLOW) - per_layer[i] * w[0]) / S(DZ);
    for (std::size_t j = 0; j < w.size(); ++j) *it++ = profit * w[j] * S(0.01) + E * S(0.001);
    return it;
  }
};

template <class S>
struct Runner {
  using value_type = S;
  odelia::ode::Solver<PstarOde<S>> solver;
  std::unique_ptr<xad::Tape<double>> tape;
  int n_steps;

  Runner(PstarOde<S> sys, int n_steps_)
      : solver(sys, odelia::ode::OdeControl()), n_steps(n_steps_) {}

  PstarOde<S>& get_system_ref() { return solver.get_system_ref(); }
  std::vector<S*> ad_parameters() { return solver.get_system_ref().ad_parameters(); }
  std::vector<S*> ad_initial_state() { return solver.get_system_ref().ad_initial_state(); }
  void reset() { solver.reset(); }
  void run() { solver.advance_fixed(grid(0.0, 1.0, n_steps)); }
};

template <class S>
S final_biomass(PstarOde<S>& sys) {
  std::vector<S> st(sys.ode_size());
  sys.ode_state(st.begin());
  return st[sys.theta.size()];
}

double run_double(double kmax, double c, int use_pstar, int nest, int nlayer, int n_steps,
                  int persist, int nfields, int graft) {
  Runner<double> r{PstarOde<double>(kmax, c, use_pstar, nest, nlayer, persist, nfields, graft), n_steps};
  r.reset();
  r.run();
  return final_biomass(r.get_system_ref());
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List pstar_ode_reprex(double kmax = 0.05, double c = 3.0, int use_pstar = 1,
                            int nest = 1, int nlayer = 2, int n_steps = 20,
                            int persist = 0, int nfields = 0, int graft = 0) {
  using adS = xad::adj<double>::active_type;

  odelia::ode::DifferentiationTargets targets;
  targets.params = {0, 1};
  targets.values = {kmax, c};

  Runner<adS> r_rev{PstarOde<adS>(adS(kmax), adS(c), use_pstar, nest, nlayer, persist, nfields, graft),
                    n_steps};
  auto [value_rev, g] = odelia::ode::compute_gradient(
      r_rev, targets, [](Runner<adS>& r) -> adS { return final_biomass(r.get_system_ref()); });

  const double hk = 1e-6 * (std::abs(kmax) + 1.0);
  const double hc = 1e-6 * (std::abs(c) + 1.0);
  const double gk_fd = (run_double(kmax + hk, c, use_pstar, nest, nlayer, n_steps, persist, nfields, graft) -
                        run_double(kmax - hk, c, use_pstar, nest, nlayer, n_steps, persist, nfields, graft)) / (2 * hk);
  const double gc_fd = (run_double(kmax, c + hc, use_pstar, nest, nlayer, n_steps, persist, nfields, graft) -
                        run_double(kmax, c - hc, use_pstar, nest, nlayer, n_steps, persist, nfields, graft)) / (2 * hc);

  // The tape is still populated here, so the recorded size per ODE step is a
  // measurement off the same run that produced the gradient. Recording this
  // composition (a central-difference stationarity residual whose evaluations each
  // rebuild the nested nodes) costs far more per step than a single node does.
  const double mem = r_rev.tape ? static_cast<double>(r_rev.tape->getMemory()) : 0.0;
  const double ops = r_rev.tape ? static_cast<double>(r_rev.tape->getNumOperations()) : 0.0;
  const double stmts = r_rev.tape ? static_cast<double>(r_rev.tape->getNumStatements()) : 0.0;

  return Rcpp::List::create(
      Rcpp::Named("value") = value_rev,
      Rcpp::Named("value_double") = run_double(kmax, c, use_pstar, nest, nlayer, n_steps, persist, nfields, graft),
      Rcpp::Named("mem_bytes") = mem,
      Rcpp::Named("ops") = ops,
      Rcpp::Named("stmts") = stmts,
      Rcpp::Named("grad_kmax") = g[0],
      Rcpp::Named("grad_c") = g[1],
      Rcpp::Named("grad_kmax_fd") = gk_fd,
      Rcpp::Named("grad_c_fd") = gc_fd,
      Rcpp::Named("use_pstar") = use_pstar,
      Rcpp::Named("nest") = nest,
      Rcpp::Named("nlayer") = nlayer,
      Rcpp::Named("n_steps") = n_steps,
      Rcpp::Named("persist") = persist,
      Rcpp::Named("nfields") = nfields,
      Rcpp::Named("graft") = graft);
}
