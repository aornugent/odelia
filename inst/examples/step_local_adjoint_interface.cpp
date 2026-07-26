/* SPIKE: does a step-local reverse sweep over a stored trajectory reproduce the
 * whole-run gradient, on a GROWING-dimension System?
 *
 * The point of this file is to settle by execution what docs/v3-step-local-adjoint.md
 * argues: that walking the run backwards one unit at a time, re-recording each unit from
 * a stored state, gives the same gradient as one whole-run tape -- and that peak tape
 * then stops growing with the run.
 *
 * DELIBERATELY NO CONTRACT. The backward loop lives here, not in odelia, and it drives
 * the Solver through members that already exist (set_ode_state / set_state_from_system /
 * advance_fixed). So this commits no interface and the boundary stays free to move: the
 * same oracle can be re-run against any candidate placement of the loop.
 *
 * Two things are parameterised because they are exactly what the design is unsure of:
 *
 *   unit_kind  - what one unit is. `step` = one ODE step; `segment` = everything between
 *                two introductions. The design retracted `segment` on the argument that
 *                its size is a schedule property; this measures both.
 *   ic_kind    - what a newborn's initial condition depends on. `constant` = 1.0, as the
 *                existing toys have it; `coupled` = c * sum(standing state), which is the
 *                shape plant actually has (a newborn's density reads the active stand).
 *                The coupled case is the falsification test: it is only right if the
 *                structural change is recorded INSIDE a unit rather than between units.
 *
 * Oracles: a closed form (constant IC only), a re-integrating central FD, and the
 * whole-run reverse tape. The FD shares no machinery with either AD path.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>

#include <vector>
#include <cmath>
#include <string>

using namespace Rcpp;

namespace {

// Cohorts decaying at a shared rate k: dy_i/dt = -k y_i, with cohorts introduced
// mid-run. The introduction is the structural change; its IC either ignores the stand
// (constant) or reads it (coupled), which is the difference that decides where the
// change must sit relative to a unit boundary.
template <class S>
class Toy {
public:
  using value_type = S;
  S k;
  double couple = 0.0;  // 0 => newborn IC is 1.0; else IC = couple * sum(y)
  std::vector<S> y;
  double time = 0.0;

  explicit Toy(S k_, double couple_ = 0.0) : k(k_), couple(couple_) { reset(); }

  std::size_t ode_size() const { return y.size(); }
  double ode_time() const { return time; }
  void reset() { y.assign(1, S(1.0)); time = 0.0; }

  // The structural change. Records nothing itself: the caller decides when it happens,
  // and on the active pass it runs ON TAPE so a coupled IC carries its adjoint.
  void introduce() {
    if (couple == 0.0) {
      y.push_back(S(1.0));
    } else {
      S total(0.0);
      for (auto const& yi : y) total += yi;
      y.push_back(couple * total);
    }
  }

  template <class It> It set_ode_state(It it, double t) {
    for (auto& yi : y) yi = *it++;
    time = t;
    return it;
  }
  template <class It> It ode_state(It it) const {
    for (auto const& yi : y) *it++ = yi;
    return it;
  }
  template <class It> It ode_rates(It it) const {
    for (auto const& yi : y) *it++ = -k * yi;
    return it;
  }
};

// ---- the run plan --------------------------------------------------------------
// One entry per unit: the times it integrates, and whether a cohort is introduced at
// its start. This is the only description of the run either pass uses, which is what
// keeps "what a unit is" a parameter rather than a structural assumption.
struct unit {
  std::vector<double> times;   // {t_from, ..., t_to}; one ODE step if size()==2
  bool introduce_at_start = false;
};

std::vector<double> grid(double a, double b, int n) {
  std::vector<double> g;
  for (int i = 0; i <= n; ++i) g.push_back(a + (b - a) * i / n);
  return g;
}

// Three segments [0,1], [1,2], [2,3] with an introduction opening the second and third.
// `by_step` splits every segment into its individual ODE steps, so the introduction
// opens a unit either way -- the two unit kinds differ only in how much each unit holds.
std::vector<unit> plan(int nstep, bool by_step) {
  std::vector<unit> u;
  for (int seg = 0; seg < 3; ++seg) {
    const std::vector<double> g = grid(seg, seg + 1, nstep);
    const bool intro = (seg > 0);
    if (!by_step) {
      u.push_back({g, intro});
    } else {
      for (std::size_t i = 0; i + 1 < g.size(); ++i)
        u.push_back({{g[i], g[i + 1]}, intro && i == 0});
    }
  }
  return u;
}

// ---- forward pass, double: the value and the stored trajectory ------------------
// Stores each unit's PRE-change state, so the structural change belongs to the unit
// that follows it and is re-recorded with it.
struct trajectory {
  std::vector<std::vector<double>> y_at;  // y_at[u] = state entering unit u, pre-change
  double value = 0.0;
};

trajectory forward(double k, double couple, const std::vector<unit>& units,
                   bool intro_inside = true) {
  Toy<double> toy(k, couple);
  odelia::ode::Solver<Toy<double>> solver(toy, odelia::ode::OdeControl());
  solver.reset();
  trajectory tr;
  for (auto const& u : units) {
    auto& sys = solver.get_system_ref();
    // intro_inside: snapshot BEFORE the change, so the change is re-recorded with the
    // unit. Otherwise snapshot AFTER it, which is the placement under test -- the
    // change then falls between units and is recorded by neither.
    if (intro_inside) {
      std::vector<double> y(sys.ode_size());
      sys.ode_state(y.begin());
      tr.y_at.push_back(y);
    }
    if (u.introduce_at_start) { sys.introduce(); solver.set_state_from_system(); }
    if (!intro_inside) {
      std::vector<double> y(sys.ode_size());
      sys.ode_state(y.begin());
      tr.y_at.push_back(y);
    }
    solver.advance_fixed(u.times);
  }
  auto& sys = solver.get_system_ref();
  std::vector<double> st(sys.ode_size());
  sys.ode_state(st.begin());
  for (auto const& v : st) tr.value += v;
  return tr;
}

// ---- whole-run reverse: the AD oracle ------------------------------------------
double whole_run_gradient(double k, double couple, const std::vector<unit>& units,
                          std::size_t* peak_bytes) {
  using ad = xad::adj<double>;
  using S = ad::active_type;
  ad::tape_type tape;
  S kk(k);
  tape.registerInput(kk);
  tape.newRecording();

  Toy<S> toy(kk, couple);
  odelia::ode::Solver<Toy<S>> solver(toy, odelia::ode::OdeControl());
  solver.reset();
  solver.get_system_ref().k = kk;
  for (auto const& u : units) {
    auto& sys = solver.get_system_ref();
    if (u.introduce_at_start) { sys.introduce(); solver.set_state_from_system(); }
    solver.advance_fixed(u.times);
  }
  auto& sys = solver.get_system_ref();
  std::vector<S> st(sys.ode_size());
  sys.ode_state(st.begin());
  S m(0.0);
  for (auto const& v : st) m += v;
  tape.registerOutput(m);
  xad::derivative(m) = 1.0;
  if (peak_bytes) *peak_bytes = tape.getMemory();
  tape.computeAdjoints();
  return xad::derivative(kk);
}

// ---- step-local reverse: the thing under test -----------------------------------
// Walks units backwards. Each unit gets its OWN tape, which is the crudest way to make
// the peak-tape claim honest (a fresh tape cannot inherit anything from the last one);
// reusing one tape with resetTo is an optimisation to try once this is proven.
double step_local_gradient(double k, double couple, const std::vector<unit>& units,
                           const trajectory& tr, std::size_t* peak_bytes,
                           bool intro_inside = true) {
  using ad = xad::adj<double>;
  using S = ad::active_type;

  const std::size_t U = units.size();
  // Seed: J = sum of final states, so every final-state adjoint is 1 and there is no
  // direct dJ/dk term.
  std::vector<double> lambda;
  {
    Toy<double> probe(k, couple);
    odelia::ode::Solver<Toy<double>> s(probe, odelia::ode::OdeControl());
    s.reset();
    for (auto const& u : units) {
      auto& sys = s.get_system_ref();
      if (u.introduce_at_start) { sys.introduce(); s.set_state_from_system(); }
      s.advance_fixed(u.times);
    }
    lambda.assign(s.get_system_ref().ode_size(), 1.0);
  }

  double grad = 0.0;
  std::size_t peak = 0;

  for (std::size_t u = U; u-- > 0;) {
    ad::tape_type tape;
    const std::vector<double>& y0 = tr.y_at[u];

    // The unit's declared inputs: its entering state and the parameter.
    std::vector<S> y_in(y0.size());
    for (std::size_t i = 0; i < y0.size(); ++i) y_in[i] = S(y0[i]);
    S kk(k);
    for (auto& yi : y_in) tape.registerInput(yi);
    tape.registerInput(kk);
    tape.newRecording();

    // Re-record exactly this unit from the stored state: the structural change first,
    // ON TAPE, then the integration.
    Toy<S> toy(kk, couple);
    toy.y.assign(y_in.begin(), y_in.end());
    toy.time = units[u].times.front();
    odelia::ode::Solver<Toy<S>> solver(toy, odelia::ode::OdeControl());
    solver.get_system_ref().k = kk;
    solver.get_system_ref().y.assign(y_in.begin(), y_in.end());
    solver.set_state_from_system();
    if (intro_inside) {
      auto& sys = solver.get_system_ref();
      if (units[u].introduce_at_start) { sys.introduce(); solver.set_state_from_system(); }
    }
    solver.advance_fixed(units[u].times);

    auto& sys = solver.get_system_ref();
    std::vector<S> out(sys.ode_size());
    sys.ode_state(out.begin());
    // With the change placed OUTSIDE the unit, the incoming lambda is one entry wider
    // than this unit's output at every introduction boundary, because the newborn came
    // into existence between units and so belongs to neither. The naive bridge -- and the
    // only one available without re-recording the change -- is to drop the newborn's
    // adjoint. That is exactly the coupling term a stand-dependent IC carries, which is
    // what makes this placement wrong rather than merely awkward.
    if (out.size() != lambda.size()) {
      if (!intro_inside && lambda.size() == out.size() + 1) {
        lambda.pop_back();
      } else {
        Rcpp::stop("unit output width != incoming lambda");
      }
    }
    for (std::size_t i = 0; i < out.size(); ++i) tape.registerOutput(out[i]);
    for (std::size_t i = 0; i < out.size(); ++i) xad::derivative(out[i]) = lambda[i];

    if (tape.getMemory() > peak) peak = tape.getMemory();
    tape.computeAdjoints();

    // The adjoint crossing the boundary is plain double, so a tape cannot span units.
    std::vector<double> lambda_prev(y_in.size());
    for (std::size_t i = 0; i < y_in.size(); ++i)
      lambda_prev[i] = xad::derivative(y_in[i]);
    if (!intro_inside && u > 0 && tr.y_at[u - 1].size() < lambda_prev.size())
      lambda_prev.resize(tr.y_at[u - 1].size());
    grad += xad::derivative(kk);   // accumulate: XAD zeroes derivatives per recording
    lambda = lambda_prev;
  }

  if (peak_bytes) *peak_bytes = peak;
  return grad;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List step_local_demo(double k = 0.3, int nstep = 10,
                           std::string unit_kind = "step",
                           std::string ic_kind = "constant",
                           std::string intro_placement = "inside",
                           double delta = 1e-5) {
  const bool by_step = (unit_kind == "step");
  const double couple = (ic_kind == "coupled") ? 0.25 : 0.0;
  const std::vector<unit> units = plan(nstep, by_step);

  const bool inside = (intro_placement == "inside");
  const trajectory tr = forward(k, couple, units, inside);

  std::size_t whole_bytes = 0, local_bytes = 0;
  const double g_whole = whole_run_gradient(k, couple, units, &whole_bytes);
  const double g_local =
      step_local_gradient(k, couple, units, tr, &local_bytes, inside);

  const double fd = (forward(k + delta, couple, units).value -
                     forward(k - delta, couple, units).value) / (2.0 * delta);

  // Closed form, constant IC only: cohorts born at t=0,1,2 with y=1, summed at t=3.
  const double analytic =
      (couple == 0.0)
          ? -(3 * std::exp(-3 * k) + 2 * std::exp(-2 * k) + std::exp(-k))
          : NA_REAL;

  // The stored trajectory, for the storage-vs-tape ratio the design rests on.
  std::size_t traj_doubles = 0;
  for (auto const& y : tr.y_at) traj_doubles += y.size();

  return Rcpp::List::create(
      Rcpp::Named("unit_kind") = unit_kind,
      Rcpp::Named("ic_kind") = ic_kind,
      Rcpp::Named("intro_placement") = intro_placement,
      Rcpp::Named("units") = static_cast<int>(units.size()),
      Rcpp::Named("value") = tr.value,
      Rcpp::Named("grad_whole_run") = g_whole,
      Rcpp::Named("grad_step_local") = g_local,
      Rcpp::Named("grad_fd") = fd,
      Rcpp::Named("grad_analytic") = analytic,
      Rcpp::Named("peak_bytes_whole_run") = static_cast<double>(whole_bytes),
      Rcpp::Named("peak_bytes_step_local") = static_cast<double>(local_bytes),
      Rcpp::Named("trajectory_bytes") = static_cast<double>(8 * traj_doubles));
}
