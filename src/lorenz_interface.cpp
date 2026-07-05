/* Lorenz system interface for odelia package
 * 
 * This file provides Lorenz-specific functions:
 * - LorenzSystem creation and manipulation
 * - Solver creation (calls generic templates with LorenzSystem type)
 * 
 * Most Solver_* functions are thin wrappers that call generic implementations
 * from ode_interface.cpp with LorenzSystem types.
 */

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/ode_fit.hpp>
#include <odelia/solver_interface.hpp>
#include <examples/lorenz_system.hpp>

using namespace Rcpp;
using namespace odelia;

// Define types for Lorenz system
typedef LorenzSystem<double> SystemType;
typedef LorenzSystem<xad::adj<double>::active_type> ActiveSystemType;

// Helper to get system pointer
inline Rcpp::XPtr<SystemType> get_system(SEXP xp) {
  return Rcpp::XPtr<SystemType>(xp);
}

//-------------------------------------------------------------------------
// System interface (Lorenz-specific)

// [[Rcpp::export]]
SEXP System_new(double sigma, double R, double b) {
  return Rcpp::XPtr<SystemType>(new SystemType(sigma, R, b), true);
}

// [[Rcpp::export]]
Rcpp::NumericVector System_pars(SEXP system_xp) {
  return Rcpp::wrap(get_system(system_xp)->pars());
}

// [[Rcpp::export]]
void System_set_params(SEXP system_xp, Rcpp::NumericVector params) {
  auto lor = get_system(system_xp);
  std::vector<double> tmp(params.begin(), params.end());
  lor->set_params(tmp.begin());
  lor->compute_rates();
}

// [[Rcpp::export]]
void System_set_state(SEXP system_xp, Rcpp::NumericVector y, double time) {
  auto lor = get_system(system_xp);
  if (y.size() != lor->ode_size()) {
    Rcpp::stop("State vector size mismatch");
  }
  std::vector<double> tmp(y.begin(), y.end());
  lor->set_ode_state(tmp.begin(), time);
}

// [[Rcpp::export]]
Rcpp::NumericVector System_state(SEXP system_xp) {
  auto lor = get_system(system_xp);
  std::vector<double> tmp(lor->ode_size());
  lor->ode_state(tmp.begin());
  return Rcpp::wrap(tmp);
}

// [[Rcpp::export]]
void System_set_initial_state(SEXP system_xp, Rcpp::NumericVector y, double t0 = 0.0) {
  auto lor = get_system(system_xp);
  if (y.size() != lor->ode_size()) {
    Rcpp::stop("State vector size mismatch");
  }
  std::vector<double> tmp(y.begin(), y.end());
  lor->set_initial_state(tmp.begin(), t0);
}

// [[Rcpp::export]]
void System_reset(SEXP system_xp) {
  auto lor = get_system(system_xp);
  lor->reset();
}

// [[Rcpp::export]]
Rcpp::NumericVector System_rates(SEXP system_xp) {
  auto lor = get_system(system_xp);
  std::vector<double> tmp(lor->ode_size());
  lor->ode_rates(tmp.begin());
  return Rcpp::wrap(tmp);
}

//-------------------------------------------------------------------------
// Solver interface (Lorenz-specific creation, generic operations)

// Solver creation - Lorenz-specific (must know LorenzSystem type). R only ever
// holds the double solver (RIF-1); gradients build the active replay internally.
// [[Rcpp::export]]
SEXP Solver_new(SEXP system_xp, SEXP control_xp) {
  Rcpp::XPtr<SystemType> sys(system_xp);
  Rcpp::XPtr<ode::OdeControl> ctrl(control_xp);
  SystemType sys_copy(*sys);
  auto* solver = new ode::Solver<SystemType>(sys_copy, *ctrl);
  return Rcpp::XPtr<ode::Solver<SystemType>>(solver, true);
}

// All other Solver functions call the generic (double-only) templates.

// [[Rcpp::export]]
void Solver_reset(SEXP solver_xp) {
  odelia::solver::Solver_reset_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
double Solver_time(SEXP solver_xp) {
  return odelia::solver::Solver_time_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
Rcpp::NumericVector Solver_state(SEXP solver_xp) {
  return odelia::solver::Solver_state_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
Rcpp::NumericVector Solver_times(SEXP solver_xp) {
  return odelia::solver::Solver_times_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
void Solver_set_state(SEXP solver_xp, Rcpp::NumericVector y, double time) {
  odelia::solver::Solver_set_state_impl<SystemType>(solver_xp, y, time);
}

// [[Rcpp::export]]
void Solver_advance_adaptive(SEXP solver_xp, Rcpp::NumericVector times) {
  odelia::solver::Solver_advance_adaptive_impl<SystemType>(solver_xp, times);
}

// [[Rcpp::export]]
void Solver_advance_fixed(SEXP solver_xp, Rcpp::NumericVector times) {
  odelia::solver::Solver_advance_fixed_impl<SystemType>(solver_xp, times);
}

// [[Rcpp::export]]
void Solver_advance_euler(SEXP solver_xp, Rcpp::NumericVector times) {
  odelia::solver::Solver_advance_euler_impl<SystemType>(solver_xp, times);
}

// [[Rcpp::export]]
void Solver_step(SEXP solver_xp) {
  odelia::solver::Solver_step_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
bool Solver_get_collect(SEXP solver_xp) {
  return odelia::solver::Solver_get_collect_impl<SystemType>(solver_xp);
}

// [[Rcpp::export]]
void Solver_set_collect(SEXP solver_xp, bool x) {
  odelia::solver::Solver_set_collect_impl<SystemType>(solver_xp, x);
}

// [[Rcpp::export]]
std::size_t Solver_get_history_size(SEXP solver_xp) {
  return odelia::solver::Solver_get_history_size_impl<SystemType>(solver_xp);
}

// Helper to get column names (Lorenz-specific)
CharacterVector get_column_names() {
  return CharacterVector::create("time", "x", "y", "z", "dxdt", "dydt", "dzdt");
}

// [[Rcpp::export]]
Rcpp::DataFrame Solver_get_history_step(SEXP solver_xp, std::size_t i) {
  return odelia::solver::Solver_get_history_step_impl<SystemType>(
    solver_xp, i, get_column_names()
  );
}

// [[Rcpp::export]]
Rcpp::List Solver_get_history(SEXP solver_xp) {
  return odelia::solver::Solver_get_history_impl<SystemType>(
    solver_xp, get_column_names()
  );
}

// [[Rcpp::export]]
void Solver_set_target(SEXP solver_xp,
                      Rcpp::NumericVector times,
                      Rcpp::NumericMatrix target,
                      Rcpp::IntegerVector obs_indices) {
  odelia::solver::Solver_set_target_impl<SystemType>(
    solver_xp, times, target, obs_indices
  );
}

// value + sum-of-squares-loss gradient on the DOUBLE handle (RIF-1): the
// double-handle successor to Solver_fit; the active replay is built, used, and
// destroyed inside the call.
// [[Rcpp::export]]
Rcpp::List Solver_value_and_gradient(SEXP solver_xp,
                                     Rcpp::Nullable<Rcpp::NumericVector> ic = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> params = R_NilValue) {
  return odelia::solver::Solver_value_and_gradient_impl<SystemType, ActiveSystemType>(
    solver_xp, ic, params
  );
}

//-------------------------------------------------------------------------
// A second functional, exercising the seam generalised in ode_fit.hpp: the
// gradient of the summed final state after a fixed-grid advance, w.r.t. the
// parameters. Nothing here mentions targets or sum-of-squares -- it is a
// stand-in for an emergent-metric functional, and proves compute_gradient no
// longer hard-codes the loss.
struct sum_final_state {
  std::vector<double> times;
  template<typename Solver>
  typename Solver::value_type operator()(Solver& solver) const {
    solver.advance_fixed(times);
    typename Solver::value_type total(0.0);
    for (auto const& s : solver.state()) {
      total += s;
    }
    return total;
  }
};

// DOUBLE solver handle; the active replay is built internally.
// [[Rcpp::export]]
Rcpp::List Solver_gradient_final_state(SEXP solver_xp,
                                       Rcpp::NumericVector times,
                                       Rcpp::NumericVector params) {
  std::vector<double> t(times.begin(), times.end());
  // Seed the parameter leaves (slots 0..n_params-1).
  ode::Independents independents;
  for (int i = 0; i < params.size(); ++i) {
    independents.slots.push_back(i);
    independents.values.push_back(params[i]);
  }
  auto [value, gradient] = odelia::solver::gradient_on_double<SystemType, ActiveSystemType>(
    solver_xp, independents, sum_final_state{t});
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("gradient") = Rcpp::wrap(gradient));
}

// A multi-output functional: the whole final state after a fixed-grid advance.
// Drives compute_jacobian's row-sweep with m = ode_size outputs.
struct final_state {
  std::vector<double> times;
  template<typename Solver>
  std::vector<typename Solver::value_type> operator()(Solver& solver) const {
    solver.advance_fixed(times);
    return solver.state();
  }
};

// DOUBLE solver handle; the active replay is built internally.
// [[Rcpp::export]]
Rcpp::List Solver_jacobian_final_state(SEXP solver_xp,
                                       Rcpp::NumericVector times,
                                       Rcpp::NumericVector params) {
  std::vector<double> t(times.begin(), times.end());
  // Seed the parameter leaves (slots 0..n_params-1).
  ode::Independents independents;
  for (int i = 0; i < params.size(); ++i) {
    independents.slots.push_back(i);
    independents.values.push_back(params[i]);
  }
  // codomain = m = the full ODE state the functional returns.
  const std::size_t codomain =
    odelia::solver::get_solver<SystemType>(solver_xp)->get_system_ref().ode_size();
  auto [values, jacobian] = odelia::solver::jacobian_on_double<SystemType, ActiveSystemType>(
    solver_xp, independents, final_state{t}, codomain);

  const size_t m = jacobian.size(), n = m ? jacobian[0].size() : 0;
  Rcpp::NumericMatrix J(m, n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      J(i, j) = jacobian[i][j];
    }
  }
  return Rcpp::List::create(Rcpp::Named("values") = Rcpp::wrap(values),
                            Rcpp::Named("jacobian") = J);
}

//-------------------------------------------------------------------------
// Comparison function for deSolve

// [[Rcpp::export]]
List lorenz_rhs(double t, NumericVector state, NumericVector pars) {
  double x = state["x"], y = state["y"], z = state["z"];
  double sigma = pars["sigma"], rho = pars["rho"], beta = pars["beta"];
  
  double dx = sigma * (y - x);
  double dy = x * (rho - z) - y;
  double dz = x * y - beta * z;
  
  return List::create(NumericVector::create(dx, dy, dz));
}

//-------------------------------------------------------------------------
// Test function for parameter type conversion

// [[Rcpp::export]]
Rcpp::List test_param_types(SEXP system_xp) {
  using ad = xad::adj<double>;
  using ad_type = ad::active_type;
  using ActiveSys = LorenzSystem<ad_type>;
  
  Rcpp::XPtr<SystemType> sys(system_xp);
  
  // Get params from passive system (doubles)
  auto params = sys->pars();
  
  // Create active system using double params (should cast to ad_type)
  auto* sys_active = new ActiveSys(params[0], params[1], params[2]);
  
  // Get params back from active system
  auto params_active = sys_active->pars();
  
  // Test if gradient flows through params
  ad::tape_type tape;
  
  std::vector<ad_type> test_params(3);
  test_params[0] = params[0];
  test_params[1] = params[1]; 
  test_params[2] = params[2];
  
  tape.registerInput(test_params[0]);
  tape.newRecording();
  
  // Recreate system with registered params
  ActiveSys sys_test(test_params[0], test_params[1], test_params[2]);
  auto retrieved = sys_test.pars();
  
  ad_type result = retrieved[0] * 2.0;
  
  tape.registerOutput(result);
  xad::derivative(result) = 1.0;
  tape.computeAdjoints();
  
  double gradient = xad::derivative(test_params[0]);
  
  delete sys_active;
  
  return Rcpp::List::create(
    Rcpp::Named("passive_params") = params,
    Rcpp::Named("active_params") = params_active,
    Rcpp::Named("gradient_test") = gradient,
    Rcpp::Named("gradient_should_be") = 2.0
  );
}