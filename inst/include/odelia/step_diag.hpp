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

}  // namespace ode
}  // namespace odelia

#endif
