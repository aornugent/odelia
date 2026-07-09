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
#include <odelia/ode_fit.hpp>
#include <odelia/rcpp_interface_helpers.hpp>

namespace odelia {
namespace solver {

// Helper to get solver pointer (templated)
template<typename T>
inline Rcpp::XPtr<ode::Solver<T>> get_solver(SEXP xp) {
  return Rcpp::XPtr<ode::Solver<T>>(xp);
}

// The step-wise Solver operations. R holds only the double Solver (RIF-1), so
// these forward straight to it -- no `active` flag, no ActiveSystemType. The AD
// replay is built internally by the gradient drivers at the bottom of this file.

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
inline Rcpp::DataFrame Solver_get_history_step_impl(SEXP solver_xp, std::size_t i,
                                                    Rcpp::CharacterVector names) {
  auto solver = get_solver<SystemType>(solver_xp);
  if (i >= solver->get_history_size()) {
    Rcpp::stop("Index out of bounds");
  }
  std::vector<double> out = solver->get_history_step(i).record_step();

  Rcpp::List df_list(names.size());
  for (size_t j = 0; j < static_cast<size_t>(names.size()); ++j) {
    df_list[j] = out[j];
  }
  df_list.attr("names") = names;
  return Rcpp::DataFrame(df_list);
}

template<typename SystemType>
inline Rcpp::List Solver_get_history_impl(SEXP solver_xp, Rcpp::CharacterVector names) {
  auto solver = get_solver<SystemType>(solver_xp);
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

template<typename SystemType>
inline void Solver_set_target_impl(SEXP solver_xp,
                                   Rcpp::NumericVector times,
                                   Rcpp::NumericMatrix target,
                                   Rcpp::IntegerVector obs_indices) {
  std::vector<double> times_vec(times.begin(), times.end());

  int nrows = target.nrow(), ncols = target.ncol();
  std::vector<std::vector<double>> targets_vec(nrows);
  for (int i = 0; i < nrows; ++i) {
    targets_vec[i].resize(ncols);
    for (int j = 0; j < ncols; ++j) {
      targets_vec[i][j] = target(i, j);
    }
  }

  std::vector<size_t> obs_idx_vec(obs_indices.size());
  for (size_t i = 0; i < static_cast<size_t>(obs_indices.size()); ++i) {
    obs_idx_vec[i] = obs_indices[i] - 1;
  }

  get_solver<SystemType>(solver_xp)->set_target(times_vec, targets_vec, obs_idx_vec);
}

// ---- Gradients on the DOUBLE handle (RIF-1, RIF-3) -------------------------
// R holds only the double Solver. These helpers differentiate on an active
// replay -- the double System lifted to active (RIF-2 rebind) -- and return
// doubles; R never sees an active type or an `active` flag.
//
// The active solver is cached on the double solver OBJECT (its `active_solver`
// member, RIF-3), so an optimiser loop reuses one active solver -- tape included --
// instead of rebuilding the active system each call. Anchoring it on the object, not
// the R XPtr's `prot` slot, means a C++ caller that holds the solver as a plain
// member (plant's SCM, which never wraps it in an XPtr) shares the reuse. Reusing it
// is pure speed: only its structural config is frozen at first build; the
// trait/parameter values are re-seeded from the Independents every call, and the
// per-call semantic state (observations, or the record->replay recording) is handed
// over by the caller (see below).
template <class SystemType, class ActiveSystemType>
ode::Solver<ActiveSystemType>& active_solver(ode::Solver<SystemType>& d) {
  if (!d.active_solver) {
    d.active_solver = std::make_shared<ode::Solver<ActiveSystemType>>(
        d.get_system_ref().template rebind_from<typename ActiveSystemType::value_type>(),
        d.get_control());
  }
  return *d.active_solver;
}

template <class SystemType, class ActiveSystemType, class Functional>
std::pair<double, std::vector<double>>
gradient_on_double(SEXP solver_xp, const ode::Independents& ind, Functional&& functional) {
  auto d = get_solver<SystemType>(solver_xp);
  auto& active = active_solver<SystemType, ActiveSystemType>(*d);
  return ode::compute_gradient(active, ind, std::forward<Functional>(functional));
}

template <class SystemType, class ActiveSystemType, class Functional>
std::pair<std::vector<double>, std::vector<std::vector<double>>>
jacobian_on_double(SEXP solver_xp, const ode::Independents& ind, Functional&& functional,
                   std::size_t codomain) {
  auto d = get_solver<SystemType>(solver_xp);
  auto& active = active_solver<SystemType, ActiveSystemType>(*d);
  return ode::compute_jacobian(active, ind, std::forward<Functional>(functional), codomain);
}

// Combined value + gradient of the sum-of-squares loss from ONE recording -- the
// calibration entry (user story §6.3): an optimiser's fn/gr share the tape rather
// than each re-running it. The double-handle successor to the retired Solver_fit.
template <class SystemType, class ActiveSystemType>
Rcpp::List Solver_value_and_gradient_impl(SEXP solver_xp,
                                          Rcpp::Nullable<Rcpp::NumericVector> ic,
                                          Rcpp::Nullable<Rcpp::NumericVector> params) {
  // Lay leaves out params-first, then the ODE initial state (param i -> slot i,
  // ic j -> slot n_params + j).
  auto d = get_solver<SystemType>(solver_xp);
  ode::Independents ind;
  const int n_params = static_cast<int>(d->get_system_ref().n_params());
  if (!params.isNull()) {
    Rcpp::NumericVector v(params);
    for (int i = 0; i < v.size(); ++i) {
      ind.slots.push_back(i);
      ind.values.push_back(v[i]);
    }
  }
  if (!ic.isNull()) {
    Rcpp::NumericVector v(ic);
    for (int j = 0; j < v.size(); ++j) {
      ind.slots.push_back(n_params + j);
      ind.values.push_back(v[j]);
    }
  }
  // Calibration's per-call semantic state is the fit observations: hand them to
  // the reused active solver each call, from the immutable double solver (the
  // sum-of-squares functional scores the trajectory against them). This is the
  // observations slice of "read the recording per call" -- kept off
  // gradient_on_double, which serves emergent functionals that carry no observations.
  auto& active = active_solver<SystemType, ActiveSystemType>(*d);
  active.set_target(d->fit_times(), d->targets(), d->obs_indices());
  auto [value, gradient] = ode::compute_gradient(active, ind, ode::sum_of_squares_loss{});
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("gradient") = Rcpp::wrap(gradient));
}

} // namespace solver
} // namespace odelia

#endif
