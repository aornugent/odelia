/* RelaxationSystem interface -- the odelia-native record -> replay demonstrator
 * (ODELIA-6). Exercises the Replayable hooks and a frozen-knot interpolator on the
 * AD path, which Lorenz/leaf_thermal never touch (has_cache was false everywhere).
 *
 * Names are Relaxation_-prefixed: this TU links into the same odelia.so as the
 * Lorenz interface, so the exported symbols must not collide.
 */

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/ode_fit.hpp>
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

// The functional: replay the recorded schedule with advance_fixed and return the
// final state. Target-free -- it never mentions set_target (design: the recording
// travels from the double Solver on its own).
struct relaxation_final {
  std::vector<double> times;
  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    solver.advance_fixed(times);
    return solver.state()[0];
  }
};

// Record on a double Solver, then replay on an active Solver and differentiate the
// final state w.r.t. `gain`. `frozen = FALSE` is the resident/live path (node
// POSITIONS frozen, values recomputed active through the interpolator -- L2);
// `frozen = TRUE` is the mutant path (field VALUES loaded as constants -- L3).
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
  //    rebind, not through set_target), differentiate.
  ActiveSystemType act = rec_sys.rebind_from<xad::adj<double>::active_type>();
  act.set_recording(rec_sys.recorded_knots(), rec_sys.recorded_values(), frozen);
  ode::Solver<ActiveSystemType> active_solver(act, *ctrl);

  ode::Independents ind;
  ind.slots.push_back(0);      // slot 0 = gain
  ind.values.push_back(gain);

  auto [value, gradient] =
      ode::compute_gradient(active_solver, ind, relaxation_final{times});

  return Rcpp::List::create(
      Rcpp::Named("value") = value,
      Rcpp::Named("gradient") = Rcpp::wrap(gradient),
      Rcpp::Named("n_steps") = static_cast<int>(times.size() - 1));
}
