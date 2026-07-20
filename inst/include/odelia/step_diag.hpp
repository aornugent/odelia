// -*-c++-*-
#ifndef ODELIA_STEP_DIAG_HPP_
#define ODELIA_STEP_DIAG_HPP_

// Optional per-attempt step log for the adaptive controller: records, for every
// stepper attempt, the step's start time, the trial step size, and whether it was
// accepted (1) or rejected and retried smaller (0). Off by default and zero-cost
// when off (a single bool test). Used only for solver diagnostics -- event-sizing
// of where the controller collapses its step (near forcing kinks, member
// insertions, threshold/clamp crossings) vs genuine fast structure. Not part of
// any model or gradient path.

#include <vector>

namespace odelia {
namespace ode {

extern bool step_log_enabled;
extern std::vector<double> step_log_t;   // step start time
extern std::vector<double> step_log_h;   // trial step size
extern std::vector<int>    step_log_ok;  // 1 accepted, 0 rejected (retried)

inline void step_log_record(double t, double h, int ok) {
  if (step_log_enabled) {
    step_log_t.push_back(t);
    step_log_h.push_back(h);
    step_log_ok.push_back(ok);
  }
}

// -------------------------------------------------------------------------
// Stage-1 event classifier: per-accepted-step monitor (off by default).
//
// When enabled AND the System offers a step_monitor() hook (see
// has_step_monitor), the adaptive controller snapshots each accepted step with
// one clean RHS eval and records the System's fixed-width vector of double
// event-margin values + int branch-signature codes, alongside (t, h). The
// analysis correlates step collapse / rejection with margin sign-changes and
// signature flips to split it into removable-events vs intrinsic fast structure.
// Not part of any model or gradient path; zero cost when off (one bool test).
//
// Margins/sigs are stored row-major flattened; the per-step column counts are
// captured on the first record (constant thereafter) so R can reshape to a
// matrix. -1 means "not yet seen".
extern bool step_monitor_enabled;
extern std::vector<double> step_mon_t;        // step start time
extern std::vector<double> step_mon_h;        // accepted step size
extern std::vector<double> step_mon_margins;  // flattened, ncol_margin per step
extern std::vector<int>    step_mon_sig;      // flattened, ncol_sig per step
extern int step_mon_ncol_margin;
extern int step_mon_ncol_sig;

inline void step_monitor_record(double t, double h,
                                const std::vector<double>& margins,
                                const std::vector<int>& sig) {
  if (!step_monitor_enabled) {
    return;
  }
  if (step_mon_ncol_margin < 0) {
    step_mon_ncol_margin = static_cast<int>(margins.size());
    step_mon_ncol_sig = static_cast<int>(sig.size());
  }
  step_mon_t.push_back(t);
  step_mon_h.push_back(h);
  step_mon_margins.insert(step_mon_margins.end(), margins.begin(), margins.end());
  step_mon_sig.insert(step_mon_sig.end(), sig.begin(), sig.end());
}

}  // namespace ode
}  // namespace odelia

#endif
