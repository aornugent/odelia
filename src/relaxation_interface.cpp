/* RelaxationSystem interface -- the odelia-native record -> replay demonstrator
 * (ODELIA-6). Exercises the Replayable hooks and a frozen-knot interpolator on the
 * AD path, which Lorenz/leaf_thermal never touch (they aren't Replayable).
 *
 * Names are Relaxation_-prefixed: this TU links into the same odelia.so as the
 * Lorenz interface, so the exported symbols must not collide.
 */

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/ode_fit.hpp>
#include <odelia/solver_interface.hpp>
#include <examples/relaxation_system.hpp>

using namespace Rcpp;
using namespace odelia;

typedef RelaxationSystem<double> SystemType;
typedef RelaxationSystem<xad::adj<double>::active_type> ActiveSystemType;

// [[Rcpp::export]]
SEXP Relaxation_new(double gain, double y0 = 1.0) {
  return Rcpp::XPtr<SystemType>(new SystemType(gain, y0), true);
}

// Fully adaptive (double) run to Tmax; the finite-difference reference the replay
// gradient is checked against. Returns the final state y(Tmax).
// [[Rcpp::export]]
double Relaxation_adaptive_final(SEXP system_xp, SEXP control_xp, double Tmax) {
  Rcpp::XPtr<SystemType> sys(system_xp);
  Rcpp::XPtr<ode::OdeControl> ctrl(control_xp);
  ode::Solver<SystemType> solver(*sys, *ctrl);
  solver.advance_adaptive({0.0, Tmax});
  return solver.state()[0];
}

// The functional: a *pure reduction* (odelia#27) -- the driver replays the
// recorded schedule with advance_fixed and hands over a positioned solver; this
// just returns the final state. Observation-free and schedule-free: it carries
// no calibration data and no grid (the recording travels from the double Solver
// on its own).
struct relaxation_final {
  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    return solver.state()[0];
  }
};

// Record on a double Solver, then replay on an active Solver and differentiate the
// final state w.r.t. `gain`. `frozen` selects the variant by choosing whether
// set_recording populates the active system's L3 field cache -- it is not a System
// mode flag (odelia#28). `frozen = FALSE` is the resident/live path (L3 empty: node
// POSITIONS frozen, values recomputed active through the interpolator -- L2);
// `frozen = TRUE` is the mutant path (L3 populated: field VALUES read as constants).
// Returns value, gradient, and the recorded step count.
// [[Rcpp::export]]
Rcpp::List Relaxation_record_replay_gradient(SEXP system_xp, SEXP control_xp,
                                             double Tmax, bool frozen = false) {
  Rcpp::XPtr<SystemType> sys(system_xp);
  Rcpp::XPtr<ode::OdeControl> ctrl(control_xp);

  // 1. Record: one adaptive double pass. The recording is owned by this immutable
  //    double Solver's System.
  SystemType dbl(*sys);
  dbl.start_recording();
  ode::Solver<SystemType> dbl_solver(dbl, *ctrl);
  dbl_solver.advance_adaptive({0.0, Tmax});

  const std::vector<double> times = dbl_solver.times();
  auto& rec_sys = dbl_solver.get_system_ref();
  const double gain = rec_sys.pars();

  // 2. Replay: lift to active (RIF-2), read the recording per call (not through
  //    rebind, not through observations), differentiate.
  ActiveSystemType act = rec_sys.rebind_from<xad::adj<double>::active_type>();
  act.set_recording(rec_sys.recorded_positions(), rec_sys.recorded_values(), frozen);
  ode::Solver<ActiveSystemType> active_solver(act, *ctrl);

  ode::DifferentiationTargets ind;
  ind.slots.push_back(0);      // slot 0 = gain
  ind.values.push_back(gain);

  auto [value, gradient] =
      ode::compute_gradient(active_solver, ind, times, relaxation_final{});

  return Rcpp::List::create(
      Rcpp::Named("value") = value,
      Rcpp::Named("gradient") = Rcpp::wrap(gradient),
      Rcpp::Named("n_steps") = static_cast<int>(times.size() - 1));
}

//-------------------------------------------------------------------------
// Persistent double Solver (RIF-3). The one-shot above rebuilds the active solver
// every call; here R holds the double Solver across calls, so the active solver --
// and its tape -- are cached on it and reused, while the recording is read fresh per call.
// This is the first Replayable (has_recording) exercise of the reuse path: Lorenz
// records nothing past L1, so its cache can never go stale against a schedule; this
// one can, which is the anti-staleness assertion RIF-3 is really about.

// [[Rcpp::export]]
SEXP Relaxation_Solver_new(SEXP system_xp, SEXP control_xp) {
  Rcpp::XPtr<SystemType> sys(system_xp);
  Rcpp::XPtr<ode::OdeControl> ctrl(control_xp);
  return Rcpp::XPtr<ode::Solver<SystemType>>(
      new ode::Solver<SystemType>(*sys, *ctrl), true);
}

// One adaptive double pass, recording the schedule + node stash into the solver's
// System. Immutable thereafter until re-recorded; returns the resolved step count.
// [[Rcpp::export]]
int Relaxation_record(SEXP solver_xp, double Tmax) {
  Rcpp::XPtr<ode::Solver<SystemType>> d(solver_xp);
  d->reset();
  d->get_system_ref().start_recording();
  d->advance_adaptive({0.0, Tmax});
  return static_cast<int>(d->recorded_steps().size() - 1);
}

// Differentiate the replayed final state w.r.t. `gain`, reusing the cached active solver.
// The recording is read from the immutable double System on THIS call -- so a fresh
// Relaxation_record() is always picked up, never replayed against a stale schedule.
// [[Rcpp::export]]
Rcpp::List Relaxation_replay_gradient(SEXP solver_xp, bool frozen = false) {
  Rcpp::XPtr<ode::Solver<SystemType>> d(solver_xp);
  if (!d->has_recording()) {
    util::stop("Relaxation_replay_gradient: call Relaxation_record() first");
  }
  auto& rec = d->get_system_ref();
  const std::vector<double> times = d->recorded_steps();

  auto& active = solver::active_solver<SystemType, ActiveSystemType>(*d);
  active.get_system_ref().set_recording(rec.recorded_positions(), rec.recorded_values(), frozen);

  ode::DifferentiationTargets ind;
  ind.slots.push_back(0);      // slot 0 = gain
  ind.values.push_back(rec.pars());

  auto [value, gradient] = ode::compute_gradient(active, ind, times, relaxation_final{});
  return Rcpp::List::create(
      Rcpp::Named("value") = value,
      Rcpp::Named("gradient") = Rcpp::wrap(gradient),
      Rcpp::Named("n_steps") = static_cast<int>(times.size() - 1));
}
