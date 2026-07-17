#ifndef ODELIA_DECIDE_HPP_
#define ODELIA_DECIDE_HPP_

#include <odelia/ode_util.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace odelia {

// Records which side a value-dependent branch takes on the recording pass and
// replays those choices, in order, on the differentiated pass. So a branch
// cannot flip between the two passes -- neither a slightly shifted value nor an
// active scalar can change the control flow the tape was built on -- and the
// gradient is the one-sided derivative of the branch that was actually taken.
//
// Usage: the System holds one branch_log. On the double pass it records
// (`start_recording`, then `decide` per branch); the recording is handed to the
// active System (`replay`), which returns the recorded choices as `decide` is
// called again in the same order. `rewind` resets the read cursor at the start
// of each replay run.
//
// The choices are stored in call order, so replay must make the same `decide`
// calls in the same order -- exact when the differentiated pass replays a fixed
// step schedule. Recording across an adaptive pass, where a rejected step's rate
// evaluations must not be kept, needs those choices committed per accepted step
// (as a recorded field is); that wrapping belongs with the adaptive System.
class branch_log {
public:
  void start_recording() {
    recording_ = true;
    taken_.clear();
    cursor_ = 0;
  }
  void replay(std::vector<char> taken) {
    recording_ = false;
    taken_ = std::move(taken);
    cursor_ = 0;
  }
  void rewind() { cursor_ = 0; }

  // Recording: store the branch and take it. Replay: return the stored branch,
  // ignoring the predicate (which is now on an active or shifted value).
  bool decide(bool predicate) {
    if (recording_) {
      taken_.push_back(predicate ? 1 : 0);
      return predicate;
    }
    return taken_.at(cursor_++) != 0;
  }

  const std::vector<char>& recorded() const { return taken_; }

private:
  bool recording_ = true;
  std::vector<char> taken_;
  std::size_t cursor_ = 0;
};

// A dead read: the plain value of x, off the tape. For a quantity that only
// informs a message or a diagnostic and must never carry a derivative.
template <class T>
double diagnostic(const T& x) {
  return util::to_passive(x);
}

}  // namespace odelia

#endif
