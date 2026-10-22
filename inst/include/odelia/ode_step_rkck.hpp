// -*-c++-*-
#ifndef ODELIA_ODE_STEP_RKCK_HPP_
#define ODELIA_ODE_STEP_RKCK_HPP_

#include <algorithm>
#include <cmath>
#include <array>
#include <vector>
#include <cstddef>
#include <XAD/XAD.hpp>
#include <odelia/adjoint.hpp>
#include <odelia/ode_control.hpp>
#include <odelia/ode_interface.hpp>

namespace odelia {
namespace ode {

template <class System>
class Step {
public:
  using value_type = typename System::value_type;
  using state_type = std::vector<value_type>;
  
  // The row a step's rate evaluations solve into and the first rates a step
  // carried in are solved_row and carried_rates (ode_interface.hpp).

  void resize(size_t size_);
  size_t order() const;
  // How stiff the System's subsystem is at the end of the last step: its rates'
  // change over its state's change, from the stage at t + h to the end.
  double subsystem_stiffness = 0.0;
  // The last step's Cash-Karp error on the subsystem, whether or not it
  // substepped, and the subsystem's inputs at its stage at t + h.
  state_type subsystem_cash_karp_error, inputs_at_last_stage;
  // Cash-Karp's solution is stable for h times a real rate down to -3.735.
  static constexpr double stability_limit = 3.735;
  // `solved` is the row this step is about to create: what its rate evaluations
  // solve for goes in, in solved_row's order, and nothing is asked of the System
  // about where it is.
  //
  // Or the row an earlier run already created, where a caller hands a CONST one:
  // the stages then LOAD what that run solved instead of solving again. Which of
  // the two happens is the constness of what was handed over and nothing else --
  // the rule `solved_scope` already follows one level down -- so there is no mode
  // here to keep, and none to hold a stale answer to.
  //
  // `substeps`, where the step substeps the System's subsystem, holds them, and
  // `inputs` are the subsystem's at the step's start.
  template <class Row>
  void step(System& system, Row& solved,
            double time, double step_size,
	    state_type &y,
	    state_type &yerr,
	    const state_type &dydt_in,
	    state_type &dydt_out,
            const subsystem_substeps* substeps = nullptr,
            const state_type& inputs = {});

  // The substeps of a step's predictor: Cash-Karp on the subsystem from `y`
  // under inputs extrapolated from `u0` along `input_slope`, controlled at
  // substep_tol and no smaller than `step_size_min`, landing on every stage
  // fraction and every knot of the subsystem's forcing.
  subsystem_substeps choose_substeps(const System& system, double time,
                                     double step_size, const state_type& y,
                                     const state_type& dydt,
                                     const state_type& u0,
                                     std::vector<double> input_slope,
                                     double step_size_min) const
    requires HasSubsystem<System>;

  // The step transposed, for as many output adjoints as are handed in: one
  // recording of the whole step, swept once per output (see adjoint_batch). The
  // active System is the walk's, held across every step of one width.
  //
  // `k1` is how the run took this step's first rates where the step before
  // carried them, and null where the run derived them afresh at `time`.
  // `substeps` holds the row's substeps where the step substepped the System's
  // subsystem.
  void step_adjoint(adjoint_system<System>& active,
                    const solved_row<System>& solved,
                    const carried_rates<System>* k1,
                    double time, double step_size,
                    const state_type &y, const adjoint_batch& lambda_out,
                    adjoint_batch& lambda_in, adjoint_batch& parameter_adjoint,
                    const subsystem_substeps* substeps = nullptr);
  // The same with the first rates derived afresh, which is what a caller
  // transposing one step on its own has: no step before it to carry them.
  void step_adjoint(adjoint_system<System>& active,
                    const solved_row<System>& solved,
                    double time, double step_size, const state_type& y,
                    const adjoint_batch& lambda_out, adjoint_batch& lambda_in,
                    adjoint_batch& parameter_adjoint) {
    step_adjoint(active, solved, nullptr, time, step_size, y, lambda_out,
                 lambda_in, parameter_adjoint);
  }

  // Rate evaluations the sweeps since the last clear have recorded, counted where
  // they are recorded rather than added up as a total the loop could disagree
  // with. Six a step, whatever the output count, because the step is recorded once
  // and swept per output. A term entering once a step where it belongs once a stage
  // divides this by six, and no gradient check can see that, because a tangent and
  // a sweep apply the same multiplier.
  std::size_t recorded_rates = 0;

  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // The tableau, written once and used at whatever scalar the caller holds its
  // rates in: the forward step and the recording its transpose is taken from
  // both step through these, so the two cannot come apart -- except in the last
  // place, where the compiler contracts a * b + c into a fused multiply-add.
  // It can do so for doubles but not through the adjoint scalar's operators, so
  // under contraction (clang's default) a stage state the recording builds can
  // be an ulp from the one the step stepped through; with -ffp-contract=off the
  // two are bit-identical.
  //
  // Y_i for stage i, into `out`: y at stage 0, and y plus the combination of the
  // earlier stage rates above that. Callers pass 1..5; see stage_row for why the
  // stage-0 arms stay.
  template <class S>
  void stage_state(int i, const std::vector<S>& y,
                   const std::vector<std::vector<S>>& k, double h,
                   std::vector<S>& out) const;
  // And the state the step ends at, y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6).
  // k2 and k5 reach it only through the later stages. `out` may be `y`.
  template <class S>
  void step_end(const std::vector<S>& y, const std::vector<std::vector<S>>& k,
                double h, std::vector<S>& out) const;
  // And its error, h times the fifth- less the fourth-order weights, over
  // `out`'s length.
  template <class S>
  void step_error(const std::vector<std::vector<S>>& k, double h,
                  std::vector<S>& out) const;
  double stage_time(int i, double time, double h) const;
  const double* stage_row(int i) const;
  // stage_state for a stage count known at compile time: the same sums in the
  // same order, with the earlier-stage count a constant so the inner loop
  // unrolls, and the data pointers held rather than re-read through the nested
  // vectors after every store. stage_state dispatches here for stages 1..5.
  template <int I, class S>
  void stage_state_fixed(const std::vector<S>& y,
                         const std::vector<std::vector<S>>& k, double h,
                         std::vector<S>& out) const;

  // The substeps' relative tolerance; the absolute one is 1e-4 of it.
  static constexpr double substep_tol = 1e-9;

  // The stages and the end for a step that substeps the subsystem from its
  // inputs `u0`: the rest from the tableau, the subsystem from a predictor under
  // the recorded slope and a corrector under the line to the inputs at the stage
  // at t + h. `rates_at(i, stage, out)` evaluates stage i. Returns, at double,
  // the subsystem's error from the coupling.
  template <class Sys, class S, class RatesAt>
  std::vector<double> take_substepped_step(const Sys& sys, double time, double h,
                                           const std::vector<S>& y0,
                                           std::vector<std::vector<S>>& k,
                                           std::vector<S>& y_end,
                                           const subsystem_substeps& substeps,
                                           const std::vector<S>& u0,
                                           RatesAt&& rates_at) const;
  // The subsystem from y0's, over the substeps ending at `ends`, its first rates
  // the step's own, under `rates(u, s, out)`; `at_end(u, s)` sees it where each
  // substep ends.
  template <class S, class Rates, class AtEnd>
  void integrate_substeps(std::size_t first, const std::vector<double>& ends,
                          double h, const std::vector<S>& y0,
                          const std::vector<std::vector<S>>& k, Rates&& rates,
                          std::vector<S>& own, AtEnd&& at_end) const;
  // One substep from fraction `from` to `to` of a step of h, from `own` whose
  // rates r[0] holds, into `out`; r[1..5] take its stage rates.
  template <class S, class Rates>
  void take_substep(const std::vector<S>& own, std::vector<std::vector<S>>& r,
                    double from, double to, double h, Rates&& rates,
                    std::vector<S>& out) const;
  // The inputs extrapolated over a step of h from u0 along `input_slope`, held
  // at zero rather than changing sign.
  template <class S>
  static std::vector<S> extrapolated_inputs(const std::vector<S>& u0,
                                            const std::vector<double>& input_slope,
                                            double h);
  // The subsystem's rates at fraction u of a step of h from `time`, under inputs
  // on the line from `from` to `to`.
  template <class Sys, class S>
  static auto rates_on_line(const Sys& sys, double time, double h,
                            const std::vector<S>& from, const std::vector<S>& to) {
    return [&sys, time, h, &from, &to, line = std::vector<S>(from.size())](
             double u, const std::vector<S>& s, std::vector<S>& out) mutable {
      for (std::size_t q = 0; q < line.size(); ++q) {
        line[q] = from[q] + (to[q] - from[q]) * u;
      }
      sys.subsystem_rates(time + u * h, s, line, out);
    };
  }

  size_t size;
  std::vector<state_type> k{6};
  state_type ytmp;
  // The subsystem's part of the stage at t + h.
  state_type subsystem_at_end_stage;

  // Cash carp constants, from GSL.
  static constexpr double ah[5] = {1.0 / 5.0, 0.3, 3.0 / 5.0, 1.0, 7.0 / 8.0};
  // The stages' fractions of the step, ascending: where a predictor's substeps
  // must land.
  static constexpr std::array<double, 5> stage_fractions = [] {
    std::array<double, 5> out{};
    std::copy(std::begin(ah), std::end(ah), out.begin());
    std::sort(out.begin(), out.end());
    return out;
  }();
  // The stage at t + h, by its index in ah.
  static constexpr int stage_at_end = static_cast<int>(
    std::find(std::begin(ah), std::end(ah), 1.0) - std::begin(ah));
  static const double b21;
  static const double b3[];
  static const double b4[];
  static const double b5[];
  static const double b6[];
  static const double c1;
  static const double c3;
  static const double c4;
  static const double c6;

  // These are the differences of fifth and fourth order coefficients
  // for error estimation
  static const double ec[];
};

template <class System>
void Step<System>::resize(size_t size_) {
  size = size_;
  for (state_type& stage_rate : k) {
    stage_rate.resize(size);
  }
  ytmp.resize(size);
}

template <class System>
size_t Step<System>::order() const {
  // In GSL, comment says "FIXME: should this be 4?"
  return 5;
}

template <class System>
template <class Row>
void Step<System>::step(System& system,
                        Row& solved,
                        double time, double step_size,
                        state_type &y,
                        state_type &yerr,
                        const state_type &dydt_in,
                        state_type &dydt_out,
                        const subsystem_substeps* substeps,
                        const state_type& inputs) {
  const double h = step_size;

  // First-same-as-last: k1 is the previous step's dydt_out, so the step costs five
  // rate evaluations and a sixth at its end to hand the next step its own k1.
  //
  // A stage's rates, handed the slot it stores what it solves for into. A System
  // that solves for nothing is handed nothing and the branch compiles away.
  auto rates_at = [&](int i, const state_type& at, state_type& into) -> void {
    if constexpr (SolvesForValues<System>) {
      ode::derivs(system, at, into, stage_time(i, time, h), solved, i - 1);
    } else {
      ode::derivs(system, at, into, stage_time(i, time, h));
    }
    if constexpr (HasSubsystem<System>) {
      if (i == stage_at_end + 1) {
        const auto [first, n] = system.subsystem();
        const auto from = at.begin() + static_cast<std::ptrdiff_t>(first);
        subsystem_at_end_stage.assign(from, from + static_cast<std::ptrdiff_t>(n));
        system.subsystem_inputs(inputs_at_last_stage);
      }
    }
  };

  std::copy(dydt_in.begin(), dydt_in.end(), k[0].begin());
  std::vector<double> subsystem_error;
  if (substeps == nullptr) {
    // The stages written out, each at a compile-time stage count, so the kernel
    // inlines into this function: a run-time stage index kept it out of line, a
    // call per stage. The same kernels stage_state dispatches to, so the sweep
    // still reverses exactly this arithmetic.
    stage_state_fixed<1>(y, k, h, ytmp); rates_at(1, ytmp, k[1]);
    stage_state_fixed<2>(y, k, h, ytmp); rates_at(2, ytmp, k[2]);
    stage_state_fixed<3>(y, k, h, ytmp); rates_at(3, ytmp, k[3]);
    stage_state_fixed<4>(y, k, h, ytmp); rates_at(4, ytmp, k[4]);
    stage_state_fixed<5>(y, k, h, ytmp); rates_at(5, ytmp, k[5]);
    step_end(y, k, h, y);
  } else if constexpr (HasSubsystem<System>) {
    subsystem_error = take_substepped_step(system, time, h, y, k, y, *substeps,
                                           inputs, rates_at);
  } else {
    util::stop("Substepping needs a System with a subsystem");
  }
  // The sixth evaluation, with a slot of its own: see `solved_row`.
  if constexpr (SolvesForValues<System>) {
    ode::derivs(system, y, dydt_out, time + h, solved, end_stage);
  } else {
    ode::derivs(system, y, dydt_out, time + h);
  }

  step_error(k, h, yerr);
  // A substepped subsystem takes its error from the coupling.
  if constexpr (HasSubsystem<System>) {
    const auto [first, n] = system.subsystem();
    double rates = 0.0, states = 0.0;
    for (size_t q = 0; q < n; ++q) {
      const double rate = util::to_passive(dydt_out[first + q]) -
        util::to_passive(k[stage_at_end + 1][first + q]);
      const double state = util::to_passive(y[first + q]) -
        util::to_passive(subsystem_at_end_stage[q]);
      rates += rate * rate;
      states += state * state;
    }
    // Two equal states measure nothing; the last measure stands.
    if (states > 0.0) {
      subsystem_stiffness = std::sqrt(rates / states);
    }
    const auto from = yerr.begin() + static_cast<std::ptrdiff_t>(first);
    subsystem_cash_karp_error.assign(from, from + static_cast<std::ptrdiff_t>(n));
    if (!subsystem_error.empty()) {
      std::copy(subsystem_error.begin(), subsystem_error.end(),
                yerr.begin() +
                  static_cast<std::ptrdiff_t>(system.subsystem().first));
    }
  }
}

// The tableau row stage i's state combines the earlier stage rates with.
//
// The stage-0 entry is kept although no caller passes 0, and so are the stage-0
// arms of stage_time and stage_state. Dropping them and re-basing this table at
// stage 2 reads as tidying away unreachable code, and it is not: this function is
// pure and inlined, so the compiler may evaluate it above stage_state's own
// `i == 1` early return, and `rows[i - 2]` is then an out-of-bounds read of a
// stack array at i == 1. It costs nothing to keep the index at i - 1 and one
// entry in the table.
template <class System>
const double* Step<System>::stage_row(int i) const {
  const double* const rows[] = {&b21, b3, b4, b5, b6};
  return rows[i - 1];
}

template <class System>
double Step<System>::stage_time(int i, double time, double h) const {
  return i == 0 ? time : time + ah[i - 1] * h;
}

template <class System>
template <class S>
void Step<System>::stage_state(int i, const std::vector<S>& y,
                               const std::vector<std::vector<S>>& k, double h,
                               std::vector<S>& out) const {
  if (i == 0) {
    std::copy(y.begin(), y.end(), out.begin());
    return;
  }
  switch (i) {
    case 1: return stage_state_fixed<1>(y, k, h, out);
    case 2: return stage_state_fixed<2>(y, k, h, out);
    case 3: return stage_state_fixed<3>(y, k, h, out);
    case 4: return stage_state_fixed<4>(y, k, h, out);
    default: return stage_state_fixed<5>(y, k, h, out);
  }
}

template <class System>
template <int I, class S>
void Step<System>::stage_state_fixed(const std::vector<S>& y,
                                     const std::vector<std::vector<S>>& k,
                                     double h, std::vector<S>& out) const {
  // Stage 1 keeps its single term grouped as b21 * h * k1: h * (b21 * k1)
  // rounds differently, and the reference numbers were blessed on this one.
  // Over y's own length, which is the whole state or a subsystem's.
  const size_t n = y.size();
  if constexpr (I == 1) {
    const S* const k0 = k[0].data();
    const S* const yp = y.data();
    S* const op = out.data();
    for (size_t q = 0; q < n; ++q) {
      op[q] = yp[q] + b21 * h * k0[q];
    }
    return;
  }
  const double* const b = stage_row(I);
  const S* kp[I];
  for (int m = 0; m < I; ++m) kp[m] = k[m].data();
  const S* const yp = y.data();
  S* const op = out.data();
  for (size_t q = 0; q < n; ++q) {
    // Summed in ascending stage, then one h. Cash-Karp's rows are dense, so
    // this is a sum over every earlier stage rather than a term for the
    // immediate predecessor.
    S combination = b[0] * kp[0][q];
    for (int m = 1; m < I; ++m) {
      combination += b[m] * kp[m][q];
    }
    op[q] = yp[q] + h * combination;
  }
}

template <class System>
template <class S>
void Step<System>::step_end(const std::vector<S>& y,
                            const std::vector<std::vector<S>>& k, double h,
                            std::vector<S>& out) const {
  const S* const k0 = k[0].data();
  const S* const k2 = k[2].data();
  const S* const k3 = k[3].data();
  const S* const k5 = k[5].data();
  const S* const yp = y.data();
  S* const op = out.data();
  const size_t n = y.size();
  for (size_t q = 0; q < n; ++q) {
    const S combination = c1 * k0[q] + c3 * k2[q] + c4 * k3[q] + c6 * k5[q];
    op[q] = yp[q] + h * combination;
  }
}

template <class System>
template <class S>
void Step<System>::step_error(const std::vector<std::vector<S>>& k, double h,
                              std::vector<S>& out) const {
  const S* const k0 = k[0].data();
  const S* const k2 = k[2].data();
  const S* const k3 = k[3].data();
  const S* const k4 = k[4].data();
  const S* const k5 = k[5].data();
  for (size_t q = 0; q < out.size(); ++q) {
    out[q] = h * (ec[1] * k0[q] + ec[3] * k2[q] + ec[4] * k3[q] +
                  ec[5] * k4[q] + ec[6] * k5[q]);
  }
}

// lambda_in[m] = (d y_end / d y)^T lambda_out[m] for the one step step() takes
// from y, and the parameter rows alongside it.
//
// ONE recording spans the whole step: its six rate evaluations and the
// combination closing them. What the sweep transposes is therefore the
// arithmetic the stepper performs, and no tableau is transposed by hand. The
// stage states are intermediates of the recording, so the step costs six model
// evaluations.
//
// The recording is derivs(), which is what the forward pass calls, so no System
// writes a transpose of its own; and the parameters ride in the same recording,
// so a stage the parameters reach carries their rows too.
template <class System>
void Step<System>::step_adjoint(adjoint_system<System>& active,
                                const solved_row<System>& solved,
                                const carried_rates<System>* k1,
                                double time, double step_size,
                                const state_type &y,
                                const adjoint_batch& lambda_out,
                                adjoint_batch& lambda_in,
                                adjoint_batch& parameter_adjoint,
                                const subsystem_substeps* substeps) {
  using scalar = adjoint_scalar<double>;
  const double h = step_size;
  if (lambda_out.empty()) {
    util::stop("step_adjoint: needs at least one output adjoint");
  }
  // The recording hands the whole state buffer to the slice below, and `size` is
  // what resize() set -- so a state of another width is checked here rather than
  // in the two callers above that happen to check it.
  util::check_length(y.size(), size);

  auto whole_step = [&](auto& sys,
                        typename std::vector<scalar>::const_iterator x,
                        std::vector<scalar>& y_end) -> void {
    const std::vector<scalar> y0(x, x + static_cast<std::ptrdiff_t>(size));
    std::vector<std::vector<scalar>> rate(6, std::vector<scalar>(size));
    std::vector<scalar> stage(size);
    // k1 is re-derived at this step's own start state, as `k1` says the run took
    // it. A row is what a replay's k1 stood in, so dropping it re-solves what the
    // replay was told; and a row loaded where the run took none is a choice the
    // run never made.
    bool k1_loaded = false;
    if constexpr (SolvesForValues<System>) {
      if (k1 != nullptr) {
        ode::derivs(sys, y0, rate[0], k1->time, k1->row, end_stage);
        k1_loaded = true;
      }
    }
    if (!k1_loaded) {
      ode::derivs(sys, y0, rate[0], k1 != nullptr ? k1->time : time);
    }
    ++recorded_rates;
    auto rates_at = [&](int i, const std::vector<scalar>& at,
                        std::vector<scalar>& into) -> void {
      if constexpr (SolvesForValues<System>) {
        ode::derivs(sys, at, into, stage_time(i, time, h), std::as_const(solved),
                    i - 1);
      } else {
        ode::derivs(sys, at, into, stage_time(i, time, h));
      }
      ++recorded_rates;
    };
    if (substeps == nullptr) {
      for (int i = 1; i < 6; ++i) {
        stage_state(i, y0, rate, h, stage);
        rates_at(i, stage, rate[i]);
      }
      step_end(y0, rate, h, y_end);
    } else if constexpr (HasSubsystem<std::remove_cvref_t<decltype(sys)>>) {
      // The subsystem's inputs at the start, off the evaluation that just ran
      // there, as the run read them.
      std::vector<scalar> inputs;
      sys.subsystem_inputs(inputs);
      take_substepped_step(sys, time, h, y0, rate, y_end, *substeps, inputs,
                           rates_at);
    } else {
      util::stop("Substepping needs a System with a subsystem");
    }
  };

  ode::state_and_parameter_adjoints(active, y, lambda_out, whole_step, lambda_in,
                                    parameter_adjoint);
}

template <class System>
subsystem_substeps Step<System>::choose_substeps(
    const System& system, double time, double step_size, const state_type& y,
    const state_type& dydt, const state_type& u0, std::vector<double> input_slope,
    double step_size_min) const
  requires HasSubsystem<System> {
  const double h = step_size;
  const auto [first, n] = system.subsystem();
  util::check_length(u0.size(), n);
  const auto at = static_cast<std::ptrdiff_t>(first);
  const auto last = at + static_cast<std::ptrdiff_t>(n);
  const std::vector<double> to = extrapolated_inputs(u0, input_slope, h);
  auto rates = rates_on_line(system, time, h, u0, to);
  std::vector<double> s(y.begin() + at, y.begin() + last), next(n), error(n);
  std::vector<std::vector<double>> r(6, std::vector<double>(n));
  std::copy(dydt.begin() + at, dydt.begin() + last, r[0].begin());
  // GSL's rule, in fractions of this step and down to the solver's smallest step.
  OdeControl control(1e-4 * substep_tol, substep_tol, 1.0, 0.0, step_size_min / h,
                     1.0, 1.0);
  control.set_controller(Controller::gsl);
  std::vector<double> stops(stage_fractions.begin(), stage_fractions.end());
  for (const double knot : system.subsystem_knots(time, time + h)) {
    const double fraction = (knot - time) / h;
    if (fraction > 0.0 && fraction < 1.0) {
      stops.push_back(fraction);
    }
  }
  std::sort(stops.begin(), stops.end());
  stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
  subsystem_substeps chosen{std::move(input_slope), {}};
  double from = 0.0, substep = 1.0;
  for (const double stop : stops) {
    while (from < stop) {
      const double end = std::min(from + substep, stop);
      take_substep(s, r, from, end, h, rates, next);
      step_error(r, (end - from) * h, error);
      // Compared with the substep asked for: `end - from` rounds away from it.
      const double asked = substep;
      substep = control.adjust_step_size(n, order(), end - from, next, error, r[0]);
      if (control.step_size_shrank()) {
        if (!(substep < asked)) {
          throw util::DomainError(
            "The subsystem fails its error test at the smallest substep");
        }
        continue;
      }
      chosen.ends.push_back(end);
      from = end;
      std::swap(s, next);
      if (from < 1.0) {
        rates(from, s, r[0]);
      }
    }
  }
  return chosen;
}

template <class System>
template <class S>
std::vector<S> Step<System>::extrapolated_inputs(
    const std::vector<S>& u0, const std::vector<double>& input_slope, double h) {
  util::check_length(input_slope.size(), u0.size());
  std::vector<S> out(u0.size());
  for (std::size_t q = 0; q < u0.size(); ++q) {
    out[q] = u0[q] + h * input_slope[q];
    if ((util::to_passive(out[q]) < 0.0) != (util::to_passive(u0[q]) < 0.0)) {
      out[q] = S(0.0);
    }
  }
  return out;
}

template <class System>
template <class Sys, class S, class RatesAt>
std::vector<double> Step<System>::take_substepped_step(
    const Sys& sys, double time, double h, const std::vector<S>& y0,
    std::vector<std::vector<S>>& k, std::vector<S>& y_end,
    const subsystem_substeps& substeps, const std::vector<S>& u0,
    RatesAt&& rates_at) const {
  const state_range subsystem = sys.subsystem();
  const auto first = static_cast<std::ptrdiff_t>(subsystem.first);
  util::check_length(u0.size(), subsystem.size);
  if (substeps.ends.empty() || substeps.ends.back() != 1.0) {
    util::stop("Substeps must end at the step's end");
  }

  // Predictor: the subsystem over the substeps, its inputs extrapolated from u0
  // along the recorded slope, kept at each stage fraction.
  std::array<std::vector<S>, stage_fractions.size()> predicted;
  std::size_t kept = 0;
  std::vector<S> s(subsystem.size);
  integrate_substeps(subsystem.first, substeps.ends, h, y0, k,
                     rates_on_line(sys, time, h, u0,
                                   extrapolated_inputs(u0, substeps.input_slope, h)),
                     s, [&](double u, const std::vector<S>& at) {
                       if (kept < predicted.size() && u == stage_fractions[kept]) {
                         predicted[kept++] = at;
                       }
                     });
  if (kept != predicted.size()) {
    util::stop("Substeps must land on every stage fraction");
  }
  auto predicted_at = [&](double u) -> const std::vector<S>& {
    const auto at = std::find(stage_fractions.begin(), stage_fractions.end(), u);
    return predicted.at(static_cast<std::size_t>(at - stage_fractions.begin()));
  };

  // Stages: the rest from the tableau, the subsystem from the predictor. Each
  // evaluation leaves that stage's inputs on the System.
  std::vector<S> stage(y0.size());
  std::array<std::vector<S>, 5> stage_inputs;
  for (int i = 1; i < 6; ++i) {
    stage_state(i, y0, k, h, stage);
    const std::vector<S>& p = predicted_at(ah[i - 1]);
    std::copy(p.begin(), p.end(), stage.begin() + first);
    rates_at(i, stage, k[i]);
    sys.subsystem_inputs(stage_inputs[i - 1]);
  }

  // Corrector: the subsystem again over the same substeps, its inputs on the
  // line from u0 to those at the stage at t + h. Its end is the step's.
  const std::vector<S>& inputs_at_end = stage_inputs[stage_at_end];
  std::vector<S> end(subsystem.size);
  integrate_substeps(subsystem.first, substeps.ends, h, y0, k,
                     rates_on_line(sys, time, h, u0, inputs_at_end), end,
                     [](double, const std::vector<S>&) {});
  step_end(y0, k, h, y_end);
  std::copy(end.begin(), end.end(), y_end.begin() + first);

  // Error, at double: how far the corrector ends from the predictor, or how far
  // the stages' inputs stray from the corrector's line, weighed as step_end
  // weighs the stages, whichever is larger.
  std::vector<double> error;
  if constexpr (std::same_as<S, double>) {
    const double weight[5] = {0.0, c3, c4, 0.0, c6};
    const std::vector<S>& predicted_end = predicted_at(1.0);
    auto corrector_line = rates_on_line(sys, time, h, u0, inputs_at_end);
    std::vector<double> stage_rates(subsystem.size), line_rates(subsystem.size);
    error.assign(subsystem.size, 0.0);
    for (int i = 1; i < 6; ++i) {
      if (weight[i - 1] == 0.0) {
        continue;
      }
      const std::vector<S>& p = predicted_at(ah[i - 1]);
      sys.subsystem_rates(stage_time(i, time, h), p, stage_inputs[i - 1],
                          stage_rates);
      corrector_line(ah[i - 1], p, line_rates);
      for (std::size_t q = 0; q < subsystem.size; ++q) {
        error[q] += h * weight[i - 1] * std::abs(stage_rates[q] - line_rates[q]);
      }
    }
    for (std::size_t q = 0; q < subsystem.size; ++q) {
      error[q] = std::max(std::abs(end[q] - predicted_end[q]), error[q]);
    }
  }
  return error;
}

template <class System>
template <class S, class Rates, class AtEnd>
void Step<System>::integrate_substeps(std::size_t first,
                                      const std::vector<double>& ends, double h,
                                      const std::vector<S>& y0,
                                      const std::vector<std::vector<S>>& k,
                                      Rates&& rates, std::vector<S>& own,
                                      AtEnd&& at_end) const {
  const std::size_t width = own.size();
  const auto at = static_cast<std::ptrdiff_t>(first);
  const auto last = at + static_cast<std::ptrdiff_t>(width);
  std::vector<std::vector<S>> r(6, std::vector<S>(width));
  std::copy(y0.begin() + at, y0.begin() + last, own.begin());
  double from = 0.0;
  for (const double to : ends) {
    if (!(to > from)) {
      util::stop("Substeps must end in ascending order inside the step");
    }
    // At the step's start the subsystem's rates are the step's own first ones.
    if (from == 0.0) {
      std::copy(k[0].begin() + at, k[0].begin() + last, r[0].begin());
    } else {
      rates(from, own, r[0]);
    }
    take_substep(own, r, from, to, h, rates, own);
    at_end(to, own);
    from = to;
  }
}

template <class System>
template <class S, class Rates>
void Step<System>::take_substep(const std::vector<S>& own,
                                std::vector<std::vector<S>>& r, double from,
                                double to, double h, Rates&& rates,
                                std::vector<S>& out) const {
  const double substep = (to - from) * h;
  std::vector<S> stage(own.size());
  for (int i = 1; i < 6; ++i) {
    stage_state(i, own, r, substep, stage);
    rates(from + ah[i - 1] * (to - from), stage, r[i]);
  }
  step_end(own, r, substep, out);
}

// RKCK coefficients, from GSL
template <class System>
const double Step<System>::b21 = 1.0 / 5.0;
template <class System>
const double Step<System>::b3[] = { 3.0 / 40.0, 9.0 / 40.0 };
template <class System>
const double Step<System>::b4[] = { 0.3, -0.9, 1.2 };
template <class System>
const double Step<System>::b5[] = {
  -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0 };

template <class System>
const double Step<System>::b6[] = {
  1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
  44275.0 / 110592.0, 253.0 / 4096.0 };

template <class System>
const double Step<System>::c1 = 37.0 / 378.0;
template <class System>
const double Step<System>::c3 = 250.0 / 621.0;
template <class System>
const double Step<System>::c4 = 125.0 / 594.0;
template <class System>
const double Step<System>::c6 = 512.0 / 1771.0;

template <class System>
const double Step<System>::ec[] = {
  0.0, 37.0 / 378.0 - 2825.0 / 27648.0, 0.0,
  250.0 / 621.0 - 18575.0 / 48384.0,
  125.0 / 594.0 - 13525.0 / 55296.0,
  -277.0 / 14336.0, 512.0 / 1771.0 - 0.25 };

}
}

#endif
