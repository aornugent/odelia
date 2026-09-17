/* Generic Solver interface templates for odelia package
 *
 * This header provides templated implementations of generic Solver functions
 * that work with any System type. System-specific interfaces include this
 * header and instantiate the templates with their specific types.
 *
 * These templates are defined inline in the header to avoid linking issues.
 */

#ifndef ODELIA_SOLVER_INTERFACE_HPP_
#define ODELIA_SOLVER_INTERFACE_HPP_

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/rcpp_interface_helpers.hpp>

namespace odelia {
namespace solver {

// Helper to get solver pointer (templated)
template<typename T>
inline Rcpp::XPtr<ode::Solver<T>> get_solver(SEXP xp) {
  return Rcpp::XPtr<ode::Solver<T>>(xp);
}

// The step-wise Solver operations. R holds only the double Solver, so these
// forward straight to it.

template<typename SystemType>
inline void Solver_reset_impl(SEXP solver_xp) {
  get_solver<SystemType>(solver_xp)->reset();
}

template<typename SystemType>
inline double Solver_time_impl(SEXP solver_xp) {
  return get_solver<SystemType>(solver_xp)->time();
}

template<typename SystemType>
inline Rcpp::NumericVector Solver_state_impl(SEXP solver_xp) {
  return Rcpp::wrap(get_solver<SystemType>(solver_xp)->state());
}

template<typename SystemType>
inline Rcpp::NumericVector Solver_times_impl(SEXP solver_xp) {
  return Rcpp::wrap(get_solver<SystemType>(solver_xp)->times());
}

template<typename SystemType>
inline void Solver_set_state_impl(SEXP solver_xp, Rcpp::NumericVector y, double time) {
  std::vector<double> yy(y.begin(), y.end());
  get_solver<SystemType>(solver_xp)->set_state(yy, time);
}

template<typename SystemType>
inline void Solver_advance_adaptive_impl(SEXP solver_xp, Rcpp::NumericVector times) {
  std::vector<double> ts(times.begin(), times.end());
  get_solver<SystemType>(solver_xp)->advance_adaptive(ts);
}

template<typename SystemType>
inline void Solver_advance_fixed_impl(SEXP solver_xp, Rcpp::NumericVector times) {
  std::vector<double> ts(times.begin(), times.end());
  get_solver<SystemType>(solver_xp)->advance_fixed(ts);
}

template<typename SystemType>
inline void Solver_advance_euler_impl(SEXP solver_xp, Rcpp::NumericVector times) {
  std::vector<double> ts(times.begin(), times.end());
  get_solver<SystemType>(solver_xp)->advance_euler(ts);
}

template<typename SystemType>
inline void Solver_step_impl(SEXP solver_xp) {
  get_solver<SystemType>(solver_xp)->step();
}

template<typename SystemType>
inline bool Solver_get_collect_impl(SEXP solver_xp) {
  return get_solver<SystemType>(solver_xp)->get_collect();
}

template<typename SystemType>
inline void Solver_set_collect_impl(SEXP solver_xp, bool x) {
  get_solver<SystemType>(solver_xp)->set_collect(x);
}

template<typename SystemType>
inline std::size_t Solver_get_history_size_impl(SEXP solver_xp) {
  return get_solver<SystemType>(solver_xp)->get_history_size();
}

template<typename SystemType>
inline Rcpp::DataFrame Solver_get_history_step_impl(SEXP solver_xp, std::size_t i) {
  auto solver = get_solver<SystemType>(solver_xp);
  if (i >= solver->get_history_size()) {
    Rcpp::stop("Index out of bounds");
  }
  Rcpp::CharacterVector names = Rcpp::wrap(solver->get_system().record_colnames());
  std::vector<double> out = solver->get_history_step(i).record_step();

  Rcpp::List df_list(names.size());
  for (size_t j = 0; j < static_cast<size_t>(names.size()); ++j) {
    df_list[j] = out[j];
  }
  df_list.attr("names") = names;
  return Rcpp::DataFrame(df_list);
}

template<typename SystemType>
inline Rcpp::List Solver_get_history_impl(SEXP solver_xp) {
  auto solver = get_solver<SystemType>(solver_xp);
  Rcpp::CharacterVector names = Rcpp::wrap(solver->get_system().record_colnames());
  const int ncols = names.size();
  const size_t nrows = solver->get_history_size();
  std::vector<std::vector<double>> cols(ncols);
  for (auto& col : cols) col.reserve(nrows);

  for (size_t i = 0; i < nrows; ++i) {
    auto row = solver->get_history_step(i).record_step();
    for (int j = 0; j < ncols; ++j) {
      cols[j].push_back(row[j]);
    }
  }

  Rcpp::List out(ncols);
  for (int j = 0; j < ncols; ++j) {
    out[j] = Rcpp::NumericVector(cols[j].begin(), cols[j].end());
  }
  out.attr("names") = names;
  return Rcpp::DataFrame(out);
}

} // namespace solver
} // namespace odelia

#endif
