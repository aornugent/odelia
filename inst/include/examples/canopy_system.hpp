#ifndef CANOPY_SYSTEM_HPP_
#define CANOPY_SYSTEM_HPP_

#include <odelia/ode_solver.hpp>
#include <odelia/interpolator.hpp>
#include <XAD/XAD.hpp>
#include <vector>
#include <utility>
#include <cmath>

using namespace odelia;

// A one-state canopy that relaxes toward the light it captures -- the demonstrator
// for record -> replay, which Lorenz and leaf_thermal don't exercise.
//
//   dy/dt = -turnover * y + L(y)
//
// y is the canopy state (say, leaf area). L is the light captured, read from a
// vertical light profile over relative depth x in [0, 1]:
//
//   light_profile(x) = gain * exp(-extinction * x)         (incident light attenuated
//                                                            with depth, Beer-Lambert)
//                    + y * exp(-shade_conc * (x - 0.5)^2)  (a leaf layer at mid-canopy
//                                                            that grows with the state)
//
// The profile is built by adaptively refining a spline over depth, then read at
// ref_depth. Because the leaf-layer term moves with y, the refinement places its
// nodes differently every step -- awkward to differentiate through directly, cheap to
// record once and replay. The ODE is linear in (gain, y), so dy(T)/dgain has a clean
// finite-difference reference.
//
// So the nodes are a CHOICE the state leaves open, and a pass re-running the model
// to tape it has to make the run's choice rather than its own -- refining again
// would resolve the profile somewhere else and tape a different function. The
// adaptive pass records them against the evaluation that made them; a later pass
// loads them and reads the profile on them with the active scalar, so its derivative
// flows while the discretisation stands still. Asked for the light VALUE back
// instead, it takes that, and the derivative through the light is then zero -- which
// is the driver declaring the light exogenous on that pass.
// What one rate evaluation chose that the state leaves open: the nodes the
// refinement placed, and the light it read off them. Plain doubles, and outside the
// class on purpose -- a choice is the same object at every scalar, so the double
// pass's record hands straight to the active one and cannot carry a derivative.
struct canopy_choice {
  std::vector<double> nodes;
  double light = 0.0;
};

template <typename T = double>
class CanopySystem {
public:
  using value_type = T;

  // Every scalar's CanopySystem is one class, so assign_from reaches the
  // source's members.
  template <typename> friend class CanopySystem;

  CanopySystem(T gain_ = T(0.0), double y0_ = 1.0,
               double turnover_ = 1.0, double extinction_ = 1.0, double shade_conc_ = 40.0,
               double ref_depth_ = 0.5, double tol_ = 1e-4, int max_depth_ = 12)
    : gain(gain_), y0_init(y0_), turnover(turnover_), extinction(extinction_),
      shade_conc(shade_conc_), ref_depth(ref_depth_), tol(tol_), max_depth(max_depth_),
      t0(0.0) {
    reset();
  }

  // Copy the configuration onto another scalar so the driver can build the active
  // version; the recording is handed in per call (set_recording), not carried here.

  // The one map: the configuration, values only. The recording is handed in per
  // call (set_recording) rather than carried here, so it is not part of it.
  template <class S1>
  void assign_from(const CanopySystem<S1>& src) {
    gain       = T(xad::value(src.gain));
    y0_init    = src.y0_init;
    turnover   = src.turnover;
    extinction = src.extinction;
    shade_conc = src.shade_conc;
    ref_depth  = src.ref_depth;
    tol        = src.tol;
    max_depth  = src.max_depth;
    reset();
  }

  template <class S2>
  CanopySystem<S2> rebind_from() const {
    CanopySystem<S2> out;
    out.assign_from(*this);
    return out;
  }

  // ---- ODE interface -------------------------------------------------------
  size_t ode_size() const { return 1; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  // Set the state and build the light profile by refining a spline over depth. The
  // refinement's nodes move with y, so this is the pass that CHOOSES them.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    y = *it++;
    time = time_;
    light = refine_light();
    compute_rates();
    return it;
  }

  // The same load at an evaluation the run recorded its choices against. Recording,
  // it refines and keeps what the refinement chose; replaying, it takes the nodes
  // back and reads the profile on them, so the discretisation stands still while the
  // profile moves with the active scalar. `reuse_light` takes the light VALUE back
  // instead, which is the driver saying the light is exogenous on this pass -- its
  // derivative is then zero rather than flowing.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_, ode::recorded_stage at) {
    y = *it++;
    time = time_;
    if (loading()) {
      const canopy_choice& made = recorded(at);
      light = reuse_light ? T(made.light) : light_on(made.nodes);
    } else {
      light = refine_light();
      if (recording) keep(at);
    }
    compute_rates();
    return it;
  }

  // Turnover minus the light captured.
  void compute_rates() { dydt = light - turnover * y; }

  template <typename Iterator>
  Iterator set_initial_state(Iterator it, double t0_ = 0.0) {
    t0 = t0_;
    y0_init = xad::value(*it++);
    return it;
  }

  template <typename Iterator>
  Iterator ode_state(Iterator it) const { *it++ = y; return it; }

  template <typename Iterator>
  Iterator ode_rates(Iterator it) const { *it++ = dydt; return it; }

  void reset() {
    y = T(y0_init);
    time = t0;
    light = refine_light();
    compute_rates();
  }

  // The single differentiable input; this canopy has no seedable initial state.
  std::vector<T*> ad_parameters()    { return {&gain}; }
  std::vector<T*> ad_initial_state() { return {}; }

  // ---- The record, System to System --------------------------------------
  // The double Solver's adaptive pass fills it; the driver hands it to the active
  // System, which then loads it rather than refining again.
  void start_recording() {
    recording = true;
    history.clear();
  }
  const std::vector<std::vector<canopy_choice>>& recorded_choices() const { return history; }

  // Hand a recording over for one replay. `reuse` = take the recorded light values
  // back as constants; otherwise only the nodes come back and the profile is read on
  // them with the active scalar.
  void set_recording(std::vector<std::vector<canopy_choice>> choices, bool reuse) {
    history = std::move(choices);
    reuse_light = reuse;
    recording = false;
  }
  bool has_recording() const { return !history.empty(); }

  double pars() const { return xad::value(gain); }

private:
  // Refine over depth until the fit resolves the profile, read it at ref_depth, and
  // keep the nodes the refinement placed.
  T refine_light() {
    const interpolator::nodes_and_data<T> chosen = interpolator::refine<T>(
        [this](double x) { return light_and_slope(x); }, 0.0, 1.0, tol, 5,
        static_cast<std::size_t>(max_depth));
    chose = chosen.x;
    interpolator::hermite_interpolator<T> interp;
    interp.init(chosen.x, chosen.y, chosen.m);
    return interp.eval(ref_depth);
  }

  // And read it on nodes already chosen, so the profile responds to the parameter
  // where the discretisation does not.
  T light_on(const std::vector<double>& nodes) {
    std::vector<T> vals, slopes;
    vals.reserve(nodes.size());
    slopes.reserve(nodes.size());
    for (double x : nodes) {
      const std::pair<T, T> vs = light_and_slope(x);
      vals.push_back(vs.first);
      slopes.push_back(vs.second);
    }
    interpolator::hermite_interpolator<T> interp;
    interp.init(nodes, vals, slopes);
    return interp.eval(ref_depth);
  }

  // Written where the address says, so a rejected attempt is overwritten by the
  // retry that replaces it and nothing has to commit a step of its own.
  void keep(ode::recorded_stage at) {
    const std::size_t stage = static_cast<std::size_t>(at.stage);
    if (history.size() <= at.step) history.resize(at.step + 1);
    if (history[at.step].size() <= stage) history[at.step].resize(stage + 1);
    history[at.step][stage] = canopy_choice{chose, xad::value(light)};
  }

  // Handed a record and not filling one. This reads what the driver GAVE this
  // System, not which pass the solver is running -- the driver said which by
  // calling start_recording or set_recording.
  bool loading() const { return !recording && !history.empty(); }

  const canopy_choice& recorded(ode::recorded_stage at) const {
    return history.at(at.step).at(static_cast<std::size_t>(at.stage));
  }

  // The profile and its slope in depth, which is the pair the interpolant takes.
  // Both come off the same two exponentials, so they cannot disagree. The return
  // type is declared rather than deduced: a deduced one hands back expression
  // templates referencing the temporaries of this return statement.
  std::pair<T, T> light_and_slope(double x) const {
    using std::exp;
    const double g = x - 0.5;
    const T beam = gain * exp(-extinction * x);
    const T shade = y * exp(-shade_conc * g * g);
    return {beam + shade, -extinction * beam - 2.0 * shade_conc * g * shade};
  }

  T gain;
  double y0_init;
  double turnover, extinction, shade_conc, ref_depth, tol;
  int max_depth;
  double t0;

  T y, dydt, light;
  double time;

  // One entry per accepted step, one per addressed stage within it. Whether this
  // pass fills it or loads it is the DRIVER's, said by which of the two methods
  // above it called -- a rejected attempt rewrites its own slots, so presence
  // cannot say it.
  bool recording = false;
  std::vector<std::vector<canopy_choice>> history;
  bool reuse_light = false;
  std::vector<double> chose;   // the nodes the last refinement placed
};

#endif
