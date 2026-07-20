// Storage + R interface for the optional adaptive-controller step log
// (odelia/step_diag.hpp). Records every step attempt (start time, trial size,
// accepted/rejected) when enabled, for solver event-sizing diagnostics.
#include <Rcpp.h>
#include <odelia/step_diag.hpp>

namespace odelia {
namespace ode {
bool step_log_enabled = false;
std::vector<double> step_log_t;
std::vector<double> step_log_h;
std::vector<int>    step_log_ok;

// Stage-1 event classifier monitor storage (see step_diag.hpp).
bool step_monitor_enabled = false;
std::vector<double> step_mon_t;
std::vector<double> step_mon_h;
std::vector<double> step_mon_margins;
std::vector<int>    step_mon_sig;
int step_mon_ncol_margin = -1;
int step_mon_ncol_sig = -1;
}  // namespace ode
}  // namespace odelia

// [[Rcpp::export]]
void step_log_enable(bool on) {
  odelia::ode::step_log_enabled = on;
}

// [[Rcpp::export]]
void step_log_reset() {
  odelia::ode::step_log_t.clear();
  odelia::ode::step_log_h.clear();
  odelia::ode::step_log_ok.clear();
}

// [[Rcpp::export]]
Rcpp::DataFrame step_log_get() {
  return Rcpp::DataFrame::create(
    Rcpp::Named("t")  = odelia::ode::step_log_t,
    Rcpp::Named("h")  = odelia::ode::step_log_h,
    Rcpp::Named("ok") = odelia::ode::step_log_ok);
}

// [[Rcpp::export]]
void step_monitor_enable(bool on) {
  odelia::ode::step_monitor_enabled = on;
}

// [[Rcpp::export]]
void step_monitor_reset() {
  odelia::ode::step_mon_t.clear();
  odelia::ode::step_mon_h.clear();
  odelia::ode::step_mon_margins.clear();
  odelia::ode::step_mon_sig.clear();
  odelia::ode::step_mon_ncol_margin = -1;
  odelia::ode::step_mon_ncol_sig = -1;
}

// Returns the monitor log as a list: t, h (length n = accepted steps), and the
// row-major-flattened margins/sig reshaped to n x ncol numeric/integer matrices
// (column meanings are defined by the System's step_monitor()).
// [[Rcpp::export]]
Rcpp::List step_monitor_get() {
  const R_xlen_t n = odelia::ode::step_mon_t.size();
  const int ncm = std::max(0, odelia::ode::step_mon_ncol_margin);
  const int ncs = std::max(0, odelia::ode::step_mon_ncol_sig);

  Rcpp::NumericMatrix margins(n, ncm);
  for (R_xlen_t i = 0; i < n; ++i) {
    for (int j = 0; j < ncm; ++j) {
      margins(i, j) = odelia::ode::step_mon_margins[i * ncm + j];
    }
  }
  Rcpp::IntegerMatrix sig(n, ncs);
  for (R_xlen_t i = 0; i < n; ++i) {
    for (int j = 0; j < ncs; ++j) {
      sig(i, j) = odelia::ode::step_mon_sig[i * ncs + j];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("t")       = odelia::ode::step_mon_t,
    Rcpp::Named("h")       = odelia::ode::step_mon_h,
    Rcpp::Named("margins") = margins,
    Rcpp::Named("sig")     = sig);
}
