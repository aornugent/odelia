/* Exercises odelia::branch_log (decide) and odelia::diagnostic on a small
 * Replayable ODE system, compiled on demand by test-ad-decide.R (sourceCpp).
 *
 * A one-state relaxation whose growth rate switches branch as the state crosses
 * a threshold: dy/dt = -turnover*y + (y < thr ? gain : 2*gain). The branch is
 * value-dependent, so on the differentiated pass -- where the state is active
 * and a perturbed parameter would move the crossing -- decide replays the branch
 * schedule recorded on the double pass instead of re-evaluating it. The gradient
 * is then the one-sided, frozen-branch derivative, which a frozen-schedule
 * finite difference reproduces. diagnostic reads the final state off the tape.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/gradient.hpp>
#include <odelia/decide.hpp>

#include <vector>

using namespace Rcpp;
using namespace odelia;

template <class T>
class RelaxDecide {
public:
  using value_type = T;

  RelaxDecide(T gain_, double y0_ = 0.5, double turnover_ = 1.0, double thr_ = 1.5)
      : gain(gain_), y0_init(y0_), turnover(turnover_), thr(thr_), t0(0.0) { reset(); }

  template <class S2> using rebind = RelaxDecide<S2>;
  template <class S2> rebind<S2> rebind_from() const {
    return rebind<S2>(xad::value(gain), y0_init, turnover, thr);
  }

  std::size_t ode_size() const { return 1; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  template <class It> It set_ode_state(It it, double time_) {
    y = *it++;
    time = time_;
    // The branch is value-dependent: recorded on the double pass, replayed here.
    const bool below = log.decide(util::to_passive(y) < thr);
    dydt = -turnover * y + (below ? gain : 2.0 * gain);
    return it;
  }
  template <class It> It set_initial_state(It it, double t0_ = 0.0) {
    t0 = t0_;
    y0_init = xad::value(*it++);
    return it;
  }
  template <class It> It ode_state(It it) const { *it++ = y; return it; }
  template <class It> It ode_rates(It it) const { *it++ = dydt; return it; }

  void reset() {
    y = T(y0_init);
    time = t0;
    log.rewind();
    // An initial rate off the recorded branch (never on the differentiated path;
    // the functional reads the final state). Uses a direct branch, not decide.
    const bool below = util::to_passive(y) < thr;
    dydt = -turnover * y + (below ? gain : 2.0 * gain);
  }

  std::vector<T*> ad_parameters() { return {&gain}; }
  std::vector<T*> ad_initial_state() { return {}; }

  branch_log log;

private:
  T gain;
  double y0_init, turnover, thr, t0;
  T y, dydt;
  double time;
};

struct final_state {
  std::size_t codomain() const { return 1; }
  template <class Solver>
  typename Solver::value_type operator()(Solver& s) const {
    using value_type = typename Solver::value_type;
    std::vector<value_type> st(1);
    s.get_system_ref().ode_state(st.begin());
    return st[0];
  }
};

// Replay the recorded branch schedule at a given gain, in double; the final state.
static double replay_final(double gain, const std::vector<char>& branches,
                           const std::vector<double>& grid, const ode::OdeControl& ctrl) {
  RelaxDecide<double> sys(gain);
  sys.log.replay(branches);
  ode::Solver<RelaxDecide<double>> solver(sys, ctrl);
  solver.advance_fixed(grid);
  std::vector<double> st(1);
  solver.get_system_ref().ode_state(st.begin());
  return st[0];
}

// [[Rcpp::export]]
Rcpp::List decide_demo(double gain = 2.0, double Tmax = 3.0, int nsteps = 30) {
  ode::OdeControl ctrl;
  std::vector<double> grid;
  for (int i = 0; i <= nsteps; ++i) grid.push_back(Tmax * i / nsteps);

  // Record the branch schedule on the double pass.
  RelaxDecide<double> dbl(gain);
  dbl.log.start_recording();
  ode::Solver<RelaxDecide<double>> dbl_solver(dbl, ctrl);
  dbl_solver.advance_fixed(grid);
  const std::vector<char> branches = dbl_solver.get_system_ref().log.recorded();
  std::vector<double> st(1);
  dbl_solver.get_system_ref().ode_state(st.begin());
  const double value_double = st[0];

  // Replay + differentiate: active gain, branch schedule frozen.
  using AD = xad::adj<double>::active_type;
  RelaxDecide<AD> act = dbl_solver.get_system_ref().rebind_from<AD>();
  act.log.replay(branches);
  ode::Solver<RelaxDecide<AD>> active_solver(act, ctrl);
  active_solver.set_schedule(grid);

  ode::DifferentiationTargets targets;
  targets.params = {0};
  targets.values = {gain};
  auto [value, grad] = ode::compute_gradient(active_solver, targets, final_state{});

  // diagnostic: the final state read off the tape is a plain double.
  const double monitored = odelia::diagnostic(value);

  // Frozen-schedule finite difference: same recorded branches, gain +/- h.
  const double h = 1e-6;
  const double fd_frozen =
      (replay_final(gain + h, branches, grid, ctrl) - replay_final(gain - h, branches, grid, ctrl)) / (2 * h);

  return Rcpp::List::create(
      Rcpp::Named("value") = value,
      Rcpp::Named("value_double") = value_double,
      Rcpp::Named("grad") = grad[0],
      Rcpp::Named("fd_frozen") = fd_frozen,
      Rcpp::Named("n_branches") = static_cast<int>(branches.size()),
      Rcpp::Named("monitored") = monitored);
}
