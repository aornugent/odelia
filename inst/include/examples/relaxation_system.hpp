#ifndef RELAXATION_SYSTEM_HPP_
#define RELAXATION_SYSTEM_HPP_

#include <odelia/ode_solver.hpp>
#include <odelia/interpolator.hpp>
#include <XAD/XAD.hpp>
#include <vector>
#include <cmath>

using namespace odelia;

// The record->replay demonstrator: a Replayable System exercising the hooks and a
// frozen-knot interpolator on the AD path (Lorenz/leaf_thermal do neither).
//
//   dy/dt = -k*y + F(y; gain)
//   F     = interp(eval_point), interp = cubic spline over an ADAPTIVE node set of
//           phi(x) = gain*exp(-decay*x) + y*exp(-beta*(x-1/2)^2), x in [0,1].
//
// phi's curvature moves with the state y, so the adaptive node placement is
// parameter-dependent and varies per step -- fragile to differentiate through,
// cheap to replay. The ODE is linear in (gain, y), so dy(T)/dgain has a clean
// finite-difference oracle.
//
// One bit of state (`recording`) and one data query, has_recorded_field() = "is the
// L3 field cache populated?"; set_recording chooses whether to populate it:
//   * record (double pass): refine nodes, stash positions per step, field per stage.
//   * replay, L3 empty (resident): recompute the field on the frozen positions with
//     the active scalar, so it self-responds to the trait.
//   * replay, L3 populated (mutant): read the field per stage as double background,
//     so the derivative through it is zero by construction.
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

  // Lift the config (values only) to another scalar; the recording is read per call
  // via set_recording, not carried here, so rebind stays free of tape identity.
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

  // Frozen-replay path (mutant): read this stage's recorded field as plain double
  // background -- no rebuild, off the tape, so d(rate)/d(field) is zero. derivs
  // routes here when has_recorded_field() is true.
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

  // ---- AD input contract ---------------------------------------------------
  size_t n_params() const { return 1; }

  // Slot 0 = gain, the single differentiable input. One trait and a constant IC keep
  // the record->replay demonstration and its finite-difference oracle sharp.
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
  // Records a node POSITION set per accepted step (L2) and a field VALUE per RK
  // stage (L3). record_* act only while recording, replay_step only while replaying.
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

  // The query derivs reads to route frozen (mutant) replay: is the L3 field cache
  // populated? Guarded against the recording pass, which is still building the cache
  // and must stay on the live recompute path.
  bool has_recorded_field() const { return !recording && !field_history.empty(); }

private:
  // Replaying = not recording and has a recording to read: derived, not stored.
  bool replaying() const { return !recording && has_recording(); }

  // The field: a cubic spline over a node set, read at eval_point. Recording refines
  // the nodes adaptively; live replay rebuilds on the frozen positions with active
  // values (the fixed-node interpolator on the AD path).
  T compute_field() {
    interpolator::basic_interpolator<T> interp;
    if (replaying()) {
      std::vector<T> vals;
      vals.reserve(node_positions.size());
      for (double x : node_positions) vals.push_back(phi(x));
      interp.init(node_positions, vals);
    } else {
      // refine (double, record) runs only on the double pass; phi's curvature moves
      // with the state y, so the placement is state-dependent per step.
      interp.construct([this](double x) { return phi(x); }, 0.0, 1.0,
                       tol, 0.0, 5, static_cast<std::size_t>(max_depth));
      if (recording) node_positions = interp.get_x();  // stash for the stage-0 record hook
    }
    return interp.eval(eval_point);
  }

  T phi(double x) const {
    using std::exp;
    const double g = x - 0.5;
    return gain * exp(-decay * x) + y * exp(-beta * g * g);
  }

  T gain;
  double y0_init;
  double k, decay, beta, eval_point, tol;
  int max_depth;
  double t0;

  T y, dydt, field;
  double bg_field = 0.0;  // frozen (L3) coupling field for the current stage: double, off-tape
  double time;

  // Recording, owned here as plain System state.
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
