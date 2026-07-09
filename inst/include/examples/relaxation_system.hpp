#ifndef RELAXATION_SYSTEM_HPP_
#define RELAXATION_SYSTEM_HPP_

#include <odelia/ode_solver.hpp>
#include <odelia/interpolator.hpp>
#include <XAD/XAD.hpp>
#include <vector>
#include <cmath>

using namespace odelia;

// RelaxationSystem -- the odelia-native demonstrator for the record -> replay
// primitive (ODELIA-6). It is the plant-agnostic shrink of FF16's resident light:
// a scalar state whose rate reads a *field* built by an adaptive interpolator over
// a set of node positions, so it exercises the two things Lorenz/leaf_thermal never
// do -- the opt-in record/replay hooks (has_cache == Replayable) and a frozen-knot
// basic_interpolator carrying an active value on the AD path.
//
//   dy/dt = -k*y + F(y; gain)
//   F     = interp(eval_point), interp = cubic spline over an ADAPTIVE node set of
//           phi(x) = gain*exp(-decay*x) + y*exp(-beta*(x-1/2)^2), x in [0,1].
//
// phi's curvature depends on the live state y, so the adaptive node placement is
// state- (hence parameter-) dependent and varies step to step -- exactly the thing
// that is fragile to differentiate through and cheap to replay once recorded. The
// ODE is linear in (gain, y), so dy(T)/dgain has a clean finite-difference oracle.
//
// Three runtime modes, one System (not three types) -- selected by two flags,
// mirroring plant's save_RK45_cache / use_cached_environment:
//   * record  (adaptive double pass): refine nodes, stash their positions per step;
//   * replay-live  (resident): frozen node POSITIONS, values recomputed with the
//     active scalar so the field self-responds to the trait (self-shading);
//   * replay-frozen (mutant): the field VALUES read per RK stage as plain DOUBLE
//     background (off the tape), so the derivative through the field is zero.
template <typename T = double>
class RelaxationSystem {
public:
  using value_type = T;

  RelaxationSystem(T gain_, double y0_ = 1.0,
                   double k_ = 1.0, double decay_ = 1.0, double beta_ = 40.0,
                   double eval_point_ = 0.5, double tol_ = 1e-4, int max_depth_ = 12)
    : gain(gain_), y0_init(y0_), k(k_), decay(decay_), beta(beta_),
      eval_point(eval_point_), tol(tol_), max_depth(max_depth_), t0(0.0) {
    reset();
  }

  // RIF-2 lift: mould the config (values only) into another scalar. The recording
  // is NOT carried here -- it is read from the immutable double System per call via
  // set_recording (design ad-record-replay.md), keeping rebind tape-identity-free.
  template <class S2> using rebind = RelaxationSystem<S2>;

  template <class S2>
  rebind<S2> rebind_from() const {
    rebind<S2> out(xad::value(gain), y0_init, k, decay, beta, eval_point, tol, max_depth);
    return out;
  }

  // ---- ODE interface -------------------------------------------------------
  size_t ode_size() const { return 1; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }

  // Adaptive / replay-live path: set state at a time and (re)build the field.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    y = *it++;
    time = time_;
    field = compute_field();
    compute_rates();
    return it;
  }

  // Replay-frozen path (mutant): the field for this step's stage `index` was
  // recorded on the double pass and is read as plain DOUBLE background -- no rebuild
  // and never on the tape, so the derivative through the field is zero by
  // construction (ad-record-replay.md sec 4.1: the frozen field is simply double
  // data, not an active constant with a zeroed derivative). odelia routes here via
  // derivs when use_cached_environment is set.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, int index) {
    y = *it++;
    bg_field = field_history.at(current_step).at(index);
    compute_rates();
    return it;
  }

  // The rate reads the coupling field. Live/record: the active interpolator value,
  // whose derivative flows (self-shading). Frozen (mutant): the recorded value added
  // as plain DOUBLE background -- off the tape, so d(rate)/d(field) is structurally
  // zero without casting a double to an active constant.
  void compute_rates() {
    dydt = -k * y;
    if (use_cached_environment) dydt += bg_field;  // frozen: double background (L3)
    else                        dydt += field;     // live/record: active field (L2)
  }

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
    replay_idx = 0;
    current_step = 0;
    if (replaying && !knot_history.empty()) {
      current_knots = knot_history.front();
    }
    if (use_cached_environment && !field_history.empty()) {
      bg_field = field_history.front().front();  // frozen field at t0 (double bg, mutant)
    } else {
      field = compute_field();
    }
    compute_rates();
  }

  // ---- AD input contract (ODELIA-3a) --------------------------------------
  size_t n_params() const { return 1; }

  // Slot layout: 0 = gain, the single differentiable input. RelaxationSystem exists
  // to demonstrate record -> replay, so it deliberately has one trait and no seedable
  // initial state -- the IC is a construction-time constant (Lorenz already
  // demonstrates initial-state sensitivity). One input keeps the L1/L2 finite-
  // difference oracle and the teaching focus sharp.
  template <typename Iterator>
  void scatter(Iterator it, const std::vector<int>& slots) {
    for (int s : slots) {
      switch (s) {
        case 0: gain = *it++; break;
        default: util::stop("RelaxationSystem::scatter: only slot 0 (gain) is seedable");
      }
    }
  }

  // ---- Record / replay hooks (Replayable) ---------------------------------
  // Recorded thing is a node POSITION (per ODE step) or a field VALUE (per RK
  // stage); see design ad-record-replay.md.
  void cache_RK45_step(int stage) {
    if (!recording) return;
    if (stage == 0) {
      step_knots = last_knots;        // freeze this step's node positions (L2)
      step_field.assign(6, 0.0);
    }
    if (stage >= 0 && stage < 6) {
      step_field[stage] = xad::value(field);  // per-stage field values (L3)
    }
  }

  void cache_ode_step() {
    if (!recording) return;
    knot_history.push_back(step_knots);
    field_history.push_back(step_field);
  }

  // On the active pass, advance to the step being replayed. Live replay pins the
  // node positions to this step's recorded comb; frozen replay leaves the field to
  // the indexed set_ode_state above. A no-op on the double (recording) pass.
  void load_ode_step() {
    if (!replaying) return;
    current_step = std::min(replay_idx, knot_history.size() - 1);
    current_knots = knot_history.at(current_step);
    ++replay_idx;
  }

  // ---- Record / replay channel (System -> System) -------------------------
  // The double Solver produces the recording and is immutable thereafter; the
  // active replay reads it per call. Positions and values are plain doubles, so
  // they cross the double->active type boundary directly.
  void start_recording() {
    recording = true; replaying = false;
    knot_history.clear(); field_history.clear();
  }
  // Positions and values are plain doubles, so the recording crosses the
  // double->active type boundary directly (no rebind, no observations).
  const std::vector<std::vector<double>>& recorded_knots()  const { return knot_history; }
  const std::vector<std::vector<double>>& recorded_values() const { return field_history; }
  void set_recording(std::vector<std::vector<double>> knots,
                     std::vector<std::vector<double>> values, bool frozen) {
    knot_history = std::move(knots);
    field_history = std::move(values);
    replaying = true; recording = false;
    use_cached_environment = frozen;  // odelia's derivs reads this to pick the path
  }
  bool has_recording() const { return !knot_history.empty(); }

  double pars() const { return xad::value(gain); }

  // odelia's derivs branch (frozen field replay) reads this member.
  bool use_cached_environment = false;

private:
  // The field: a cubic spline (basic_interpolator<T>) over an adaptive node set,
  // read at eval_point. On a replay the node positions are frozen (current_knots)
  // and only the values go active -- the fixed-node interpolator on the AD path.
  T compute_field() {
    std::vector<double> knots = replaying ? current_knots : refine_knots();
    if (!replaying) last_knots = knots;   // stash for the record hook
    std::vector<T> vals;
    vals.reserve(knots.size());
    for (double x : knots) vals.push_back(phi(x));
    interpolator::basic_interpolator<T> interp;
    interp.init(knots, vals);
    return interp.eval(eval_point);
  }

  T phi(double x) const {
    using std::exp;
    const double g = x - 0.5;
    return gain * exp(-decay * x) + y * exp(-beta * g * g);
  }

  // phi in doubles, for the adaptive placement decision (never differentiated --
  // adaptivity runs on the double pass only).
  double phi_double(double x) const {
    return xad::value(gain) * std::exp(-decay * x)
         + xad::value(y) * std::exp(-beta * (x - 0.5) * (x - 0.5));
  }

  void refine(double a, double b, int depth, std::vector<double>& mids) const {
    const double m = 0.5 * (a + b);
    const double approx = 0.5 * (phi_double(a) + phi_double(b));
    if (depth < max_depth && std::fabs(phi_double(m) - approx) > tol) {
      refine(a, m, depth + 1, mids);
      mids.push_back(m);                 // in-order push keeps `mids` sorted
      refine(m, b, depth + 1, mids);
    }
  }

  std::vector<double> refine_knots() const {
    std::vector<double> mids;
    refine(0.0, 1.0, 0, mids);
    std::vector<double> knots;
    knots.reserve(mids.size() + 2);
    knots.push_back(0.0);
    knots.insert(knots.end(), mids.begin(), mids.end());
    knots.push_back(1.0);
    if (knots.size() < 3) knots = {0.0, 0.5, 1.0};  // basic_interpolator needs >= 3
    return knots;
  }

  T gain;
  double y0_init;
  double k, decay, beta, eval_point, tol;
  int max_depth;
  double t0;

  T y, dydt, field;
  double bg_field = 0.0;  // frozen (L3) coupling field: plain double background, off-tape
  double time;

  // Recording (owned here as System state; odelia grows no Recording noun).
  bool recording = false;
  bool replaying = false;
  std::vector<double> last_knots;                    // most recent adaptive comb
  std::vector<double> step_knots;                    // this step's frozen comb
  std::vector<double> step_field;                    // this step's per-stage values
  std::vector<std::vector<double>> knot_history;     // [step] node positions
  std::vector<std::vector<double>> field_history;    // [step][stage] field values
  std::size_t replay_idx = 0;
  std::size_t current_step = 0;
  std::vector<double> current_knots;
};

#endif
