/* Multirate (MRI-GARK) interface for the two-rate demonstrator.
 *
 * Exposes the model-agnostic MRI macro stepper on the TwoRateSystem example so
 * the native multirate capability can be checked from R: a macro-stepped run,
 * and a single-rate adaptive reference of the same system to check it against.
 */

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/mri.hpp>
#include <examples/two_rate_system.hpp>

using namespace Rcpp;
using namespace odelia;
using namespace odelia::ode;

static MRICoupling pick_table(const std::string& name) {
  if (name == "forward_euler") return mri_forward_euler();
  if (name == "heun")          return mri_heun();
  if (name == "kutta3")        return mri_kutta3();
  if (name == "erk33a")        return mri_erk33a();
  Rcpp::stop("unknown MRI table: " + name);
}

static Rcpp::NumericMatrix to_matrix(const std::vector<std::vector<double>>& h) {
  const int nt = (int)h.size(), nd = h.empty() ? 0 : (int)h[0].size();
  Rcpp::NumericMatrix out(nt, nd);
  for (int i = 0; i < nt; ++i)
    for (int j = 0; j < nd; ++j) out(i, j) = h[i][j];
  return out;
}

// [[Rcpp::export]]
Rcpp::List two_rate_mri(double k, int n_slow, std::string table,
                        Rcpp::NumericVector macro_times, double tol) {
  TwoRateSystem<double> sys(k, n_slow);
  MRICoupling M = pick_table(table);
  OdeControl control(tol, tol, 1.0, 0.0, 1e-10, 1e10, 1e-4);
  std::vector<double> times(macro_times.begin(), macro_times.end());
  long n_fast = 0;
  auto hist = mri_advance(sys, M, control, times, &n_fast);
  return Rcpp::List::create(_["states"] = to_matrix(hist),
                            _["n_fast"] = (double)n_fast,
                            _["order"]  = M.order);
}

// [[Rcpp::export]]
Rcpp::List two_rate_reference(double k, int n_slow, Rcpp::NumericVector times, double tol) {
  TwoRateSystem<double> sys(k, n_slow);
  OdeControl control(tol, tol, 1.0, 0.0, 1e-10, 1e10, 1e-4);
  Solver<TwoRateSystem<double>> solver(sys, control);
  std::vector<double> t(times.begin(), times.end());
  solver.advance_adaptive(t);

  auto hist = solver.get_history();
  const int nt = (int)hist.size(), nd = (int)sys.ode_size();
  Rcpp::NumericMatrix out(nt, nd);
  for (int i = 0; i < nt; ++i) {
    std::vector<double> s(nd);
    hist[i].ode_state(s.begin());
    for (int j = 0; j < nd; ++j) out(i, j) = s[j];
  }
  const long nsteps = (long)solver.times().size() - 1;
  return Rcpp::List::create(_["states"] = out, _["nsteps"] = (double)nsteps);
}
