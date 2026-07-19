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
