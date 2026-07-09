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
// do -- the opt-in record/replay hooks (the Replayable concept) and a frozen-knot
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
// There is no `live | frozen | replaying` mode flag (odelia#28). The System stores
// one bit -- `recording` -- and answers one data query -- `has_recorded_field()`,
// literally "is my L3 field cache populated?". The variant entry (`set_recording`
// with `frozen`) chooses whether to populate that cache; the System just reads what
// is present:
//   * record  (adaptive double pass): refine nodes, stash their positions per step
//     and the field value per RK stage; `recording` true.
//   * replay, L3 cache empty (resident/live): recompute the field on the frozen node
//     POSITIONS with the active scalar, so the field self-responds to the trait.
//   * replay, L3 cache populated (mutant/frozen): read the field VALUES per RK stage
//     as plain DOUBLE background (off the tape), so the derivative through the field
//     is zero by construction.
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

  // Adaptive / live-replay path: set state at a time and (re)build the field.
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    y = *it++;
    time = time_;
    field = compute_field();
    compute_rates();
    return it;
  }

  // Frozen-replay path (mutant): the field for this step's stage `index` was
  // recorded on the double pass and is read as plain DOUBLE background -- no rebuild
  // and never on the tape, so the derivative through the field is zero by
  // construction (ad-record-replay.md sec 4.1: the frozen field is simply double
  // data, not an active constant with a zeroed derivative). odelia routes here via
  // derivs when has_recorded_field() is true (the L3 cache is populated).
  template <typename Iterator>
  Iterator set_ode_state(Iterator it, int index) {
    y = *it++;
    bg_field = stage_field.at(index);
    compute_rates();
    return it;
  }

  // The rate reads the coupling field. Live/record: the active interpolator value,
  // whose derivative flows (self-shading). Frozen (mutant): the recorded value added
  // as plain DOUBLE background -- off the tape, so d(rate)/d(field) is structurally
  // zero without casting a double to an active constant.
  void compute_rates() {
    dydt = -k * y;
    if (has_recorded_field()) dydt += bg_field;  // frozen: double background (L3)
    else                      dydt += field;     // live/record: active field (L2)
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
    step = 0;
    if (has_recorded_field()) {
      stage_field = field_history.front();       // frozen: this step's per-stage values
      bg_field = stage_field.front();            // frozen field at t0 (double bg, mutant)
    } else {
      if (replaying()) node_positions = positions_history.front();  // live: seed the initial comb
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
  // The recorded thing is a node POSITION set (per accepted step, L2) and a field
  // VALUE per RK stage (L3); see design ad-record-replay.md sec 3. `record_stage` /
  // `record_ode_step` act only while recording; `replay_step` only while replaying.
  void record_stage(int stage) {
    if (!recording) return;
    if (stage == 0) {
      pending_positions = node_positions;  // freeze this step's node positions (L2, stage 0)
      stage_field.assign(6, 0.0);
    }
    if (stage >= 0 && stage < 6) {
      stage_field[stage] = xad::value(field);  // per-stage field values (L3)
    }
  }

  void record_ode_step() {
    if (!recording) return;
    positions_history.push_back(pending_positions);
    field_history.push_back(stage_field);
  }

  // On the active pass, load this step's recorded slice, then advance the single
  // cursor. Live replay reads `node_positions` (the frozen comb) when rebuilding the
  // field; frozen replay reads `stage_field` per RK stage. A no-op on the double
  // (recording) pass. The schedule matches the recording (the driver replays
  // recorded_steps()), so the cursor walks [0, size) with no clamp.
  void replay_step() {
    if (!replaying()) return;
    node_positions = positions_history.at(step);
    if (has_recorded_field()) stage_field = field_history.at(step);
    ++step;
  }

  // ---- Record / replay channel (System -> System) -------------------------
  // The double Solver produces the recording and is immutable thereafter; the
  // active replay reads it per call. Positions and values are plain doubles, so
  // they cross the double->active type boundary directly.
  void start_recording() {
    recording = true;
    positions_history.clear(); field_history.clear();
  }
  const std::vector<std::vector<double>>& recorded_positions() const { return positions_history; }
  const std::vector<std::vector<double>>& recorded_values()    const { return field_history; }

  // Hand a recording to the active system for one replay. `frozen` selects the
  // variant: a mutant/frozen replay populates the L3 field cache (read back as
  // double background); a resident/live replay leaves it empty (the field is
  // recomputed on the frozen positions with the active scalar). This is the only
  // thing that decides which path derivs takes -- has_recorded_field() then just
  // reports whether the cache is populated.
  void set_recording(std::vector<std::vector<double>> positions,
                     std::vector<std::vector<double>> values, bool frozen) {
    positions_history = std::move(positions);
    field_history = frozen ? std::move(values) : std::vector<std::vector<double>>{};
    recording = false;
    step = 0;
  }
  bool has_recording() const { return !positions_history.empty(); }

  double pars() const { return xad::value(gain); }

  // The one data query derivs reads to route frozen-field (mutant) replay: is the L3
  // field cache populated? Guarded against the recording pass, which is building that
  // cache and must stay on the live (recompute) path (odelia#28).
  bool has_recorded_field() const { return !recording && !field_history.empty(); }

private:
  // A System is replaying exactly when it is not recording and has a recording to
  // read -- derived, not a stored mode flag (design ad-record-replay.md sec 5.1).
  bool replaying() const { return !recording && has_recording(); }

  // The field: a cubic spline (basic_interpolator<T>) over a node set, read at
  // eval_point. Recording refines the nodes adaptively; live replay reuses the frozen
  // node positions and only the values go active (the fixed-node interpolator on the
  // AD path). Once the interpolator owns its own knots (odelia#22) this inline refiner
  // and `node_positions` go away.
  T compute_field() {
    std::vector<double> knots = replaying() ? node_positions : refine_knots();
    if (recording) node_positions = knots;   // stash for the stage-0 record hook
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
  double bg_field = 0.0;  // frozen (L3) coupling field for the current stage: double, off-tape
  double time;

  // Recording (owned here as System state; odelia grows no Recording noun).
  bool recording = false;                            // set only during the double pass
  std::vector<std::vector<double>> positions_history;  // L2: node positions per accepted step
  std::vector<std::vector<double>> field_history;      // L3: field values [step][stage]

  // Single replay cursor + this step's slices. `node_positions` is the comb
  // compute_field just built (recording) or the frozen comb replay_step loaded (live
  // replay); `stage_field` is the current step's per-stage values (accumulated while
  // recording, loaded for a frozen replay); `pending_positions` freezes stage 0 until
  // the step is accepted and committed.
  std::size_t step = 0;
  std::vector<double> node_positions;
  std::vector<double> stage_field;
  std::vector<double> pending_positions;
};

#endif
