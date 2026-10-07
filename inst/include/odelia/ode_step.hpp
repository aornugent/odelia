// -*-c++-*-
#ifndef ODELIA_ODE_STEP_HPP_
#define ODELIA_ODE_STEP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
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
  
  using solved_values = solved_values_t<System>;
  using solved_row = ode::solved_row<solved_values>;

  void resize(size_t size_);
  size_t order() const;
  // `solved` is the row this step creates: each of its six evaluations stores
  // into its own slot, which holds whatever the caller put there beforehand.
  // `alone`, where the step takes the System's block alone, is the row's record,
  // which the step takes as recorded, and `inputs` the block's at the start.
  void step(System& system, solved_row& solved,
            double time, double step_size,
	    state_type &y,
	    state_type &yerr,
	    const state_type &dydt_in,
	    state_type &dydt_out,
            const alone_steps* alone = nullptr,
            const state_type& inputs = {});

  // Where the predictor's inner steps end: Cash-Karp on the block alone under
  // `inputs` extrapolated by `slope`, at alone_tol, landing on every stop.
  std::vector<double> alone_ends(const System& system, double time,
                                 double step_size, const state_type& y,
                                 const state_type& dydt, const state_type& inputs,
                                 const std::vector<double>& slope,
                                 double step_size_min) const
    requires StepsBlockAlone<System>;

  // The step just taken, handed to a System that splits, which rewrites `y_end`
  // for each block it splits. The sign values are set at double only.
  template <class S>
  struct taken_step {
    const Step& stepper;
    double time;
    double h;
    const std::vector<S>& y0;
    const std::vector<std::vector<S>>& k;
    const std::vector<S>& end_rate;
    std::vector<S>& y_end;
    const std::vector<double>* sign_values_in = nullptr;
    const std::array<std::vector<double>, 6>* sign_values = nullptr;
    // Where the step took the System's block alone: the predictor's block at each
    // sample fraction, and where the block starts.
    const std::array<std::vector<S>, 5>* alone_samples = nullptr;
    std::size_t alone_first = 0;

    // Where a System samples what its blocks read. The quartic through the five
    // samples is the dense output exactly, and a smooth function of it to O(h^5).
    static constexpr std::array<double, 5> sample_fractions{0.0, 0.25, 0.5, 0.75,
                                                            1.0};

    // As many components from `first` as `out` holds, of the state at fraction u
    // of the step, on Cash-Karp's fourth-order continuous extension. A block the
    // step took alone is read from the predictor's samples instead.
    template <class U>
    void dense_state(const U& u, std::size_t first, std::vector<S>& out) const {
      stepper.dense_state(u, h, y0, k, end_rate, first, out);
      if (alone_samples == nullptr) {
        return;
      }
      std::vector<S> block;
      sample_at(u, *alone_samples, block);
      const std::size_t lo = std::max(first, alone_first);
      const std::size_t hi =
        std::min(first + out.size(), alone_first + block.size());
      for (std::size_t q = lo; q < hi; ++q) {
        out[q - first] = block[q - alone_first];
      }
    }
    // The quartic through samples taken at each of sample_fractions, at u.
    template <class U>
    void sample_at(const U& u, const std::array<std::vector<S>, 5>& samples,
                   std::vector<S>& out) const;
    // The block from `first` integrated in pieces meeting at `split_at`, fractions
    // ascending inside (0, 1); `rates(u, own, out)` gives its rates at u.
    template <class U, class Rates>
    void integrate_pieces(std::size_t first, const std::vector<U>& split_at,
                          Rates&& rates, std::vector<S>& own) const {
      stepper.integrate_pieces(first, split_at, h, y0, k, rates, own,
                               [](const U&, const std::vector<S>&) {});
    }
    // Block `block`'s sign changes: one where the ends differ, two where an inner
    // reading crosses or nears zero; `value_at(u, solved)` fills a given `solved`.
    template <class Values, class ValueAt>
    std::vector<sign_change<Values>> sign_changes(std::size_t block,
                                                  ValueAt&& value_at) const
      requires std::same_as<S, double>;
  };

  // The step just taken, which ended at `y` with rates `dydt_out`, at double.
  taken_step<double> taken(double time, double step_size,
                           const state_type& dydt_out, state_type& y,
                           const std::vector<double>& sign_values_in) const {
    return {*this, time, step_size, y_start, k, dydt_out, y, &sign_values_in,
            &sign_values, alone_samples[0].empty() ? nullptr : &alone_samples,
            alone_first};
  }
  // Each block's sign value where the step just taken ended, which the next step
  // starts from; read again after the end's rates are evaluated again.
  const std::vector<double>& end_sign_values() const { return sign_values[5]; }
  void read_end_sign_values(const System& system) {
    system.sign_values(sign_values[5]);
  }

  // The step transposed, for as many seeds as are handed in: one recording of the
  // whole step, swept once per seed. The active System is the walk's, held across
  // every step of one width. A caller wanting one row passes a batch of
  // one; there is no separate entry point for that, because a second signature
  // over the same recording is a second place for the seam between the state and
  // the parameter halves to be got wrong.
  //
  // `first` is what the evaluation at the step's start state solved for, and
  // `first_time` when it ran: the row below's `at_state`. `alone` is the row's
  // record where the step took the System's block alone.
  void step_adjoint(active_system<System>& active,
                    const solved_values& first, double first_time,
                    const solved_row& solved,
                    double time, double step_size,
                    const state_type &y, const adjoint_rows& lambda_out,
                    adjoint_rows& lambda_in, adjoint_rows& parameter_adjoint,
                    const alone_steps& alone = alone_steps{});

  // Rate evaluations the sweeps since the last clear have recorded, counted where
  // they are recorded rather than added up as a total the loop could disagree
  // with. Six a step, whatever the seed count, because the step is recorded once
  // and swept per seed. A term entering once a step where it belongs once a stage
  // divides this by six, and no gradient check can see that, because a tangent and
  // a sweep apply the same multiplier.
  std::size_t recorded_rates = 0;

  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // The five stages and the end, at the caller's scalar: the forward step and the
  // recording its transpose is taken from both take this, so the two cannot come
  // apart. k[0] holds y0's rates; `y_end` may be `y0`.
  template <class Sys, class Row, class S>
  void take_step(Sys& sys, Row& solved, double time, double h,
                 const std::vector<S>& y0, std::vector<std::vector<S>>& k,
                 std::vector<S>& y_end);

  // Where the predictor of a step taken alone stops: each stage's fraction and
  // each sample fraction, written as ah[] and sample_fractions write them.
  static constexpr std::array<double, 8> alone_stops{
    1.0 / 5.0, 0.25, 0.3, 0.5, 3.0 / 5.0, 0.75, 7.0 / 8.0, 1.0};
  // The inner steps' relative tolerance; the absolute one is 1e-4 of it.
  static constexpr double alone_tol = 1e-9;

  // A step taken alone: in, the row's record and the block's inputs at its start;
  // out, the predictor's block at the sample fractions and, at double, its error.
  template <class S>
  struct alone_pass {
    const alone_steps& record;
    const std::vector<S>& inputs;
    std::array<std::vector<S>, 5> samples{};
    std::vector<double> error{};
  };
  // take_step for a step taken alone, over the record's inner steps.
  template <class Sys, class Row, class S>
  void take_step_alone(Sys& sys, Row& solved, double time, double h,
                       const std::vector<S>& y0, std::vector<std::vector<S>>& k,
                       std::vector<S>& y_end, alone_pass<S>& alone);
  // The inputs extrapolated over the step by `slope`, stopping at zero rather
  // than changing sign.
  template <class S>
  static std::vector<S> predicted_inputs(const std::vector<S>& inputs,
                                         const std::vector<double>& slope,
                                         double h);

  // The tableau, written once and used at whatever scalar the caller holds its
  // rates in.
  //
  // Y_i for stage i, into `out`: y at stage 0, and y plus the combination of the
  // earlier stage rates above that. Callers pass 1..5; see stage_row for why the
  // stage-0 arms stay. Both run over y's length, so a block's pieces reuse them.
  template <class S, class H>
  void stage_state(int i, const std::vector<S>& y,
                   const std::vector<std::vector<S>>& k, const H& h,
                   std::vector<S>& out) const;
  // And the state the step ends at, y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6).
  // k2 and k5 reach it only through the later stages. `out` may be `y`.
  template <class S, class H>
  void step_end(const std::vector<S>& y, const std::vector<std::vector<S>>& k,
                const H& h, std::vector<S>& out) const;
  // And its error, h times the fifth- less the fourth-order weights, over
  // `out`'s length.
  template <class S>
  void step_error(const std::vector<std::vector<S>>& k, double h,
                  std::vector<S>& out) const;
  // As many components from `first` as `out` holds, of the state at fraction u
  // of a step from y0, through its stage rates k and the end's rate.
  template <class S, class U>
  void dense_state(const U& u, double h, const std::vector<S>& y0,
                   const std::vector<std::vector<S>>& k,
                   const std::vector<S>& end_rate, std::size_t first,
                   std::vector<S>& out) const;
  // `at_end(u, own)` sees the block where each piece ends.
  template <class S, class U, class Rates, class AtEnd>
  void integrate_pieces(std::size_t first, const std::vector<U>& split_at,
                        double h, const std::vector<S>& y0,
                        const std::vector<std::vector<S>>& k, Rates&& rates,
                        std::vector<S>& own, AtEnd&& at_end) const;
  double stage_time(int i, double time, double h) const;
  const double* stage_row(int i) const;

  size_t size;
  std::vector<state_type> k{6};
  // Where the step just taken started, which a System that splits reads.
  state_type y_start;
  // Each block's sign value at the five stages, then at the end, of that step.
  std::array<std::vector<double>, 6> sign_values;
  // Where that step took the System's block alone, the predictor's block at the
  // sample fractions and where the block starts; empty otherwise.
  std::array<std::vector<double>, 5> alone_samples;
  std::size_t alone_first = 0;

  // Cash carp constants, from GSL.
  static const double ah[];
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

  // The dense output's weight on each rate (k1..k6, then the end's), as the
  // coefficient of u, u^2, u^3 and u^4.
  static const double dense_weights[4][7];
};

// A System whose rates change form where a block's sign value changes sign: it
// reports the sign values, splits each step, and splits a walk's step where the
// run's split, recording the blocks the walk split.
// ⚠️ split_sign_changes IS TRUE WHEREVER IT EVALUATED, split or not: the
// System is then off the step's end, whose rates are evaluated again.
template <typename System>
concept SplitsSignChanges =
  std::same_as<typename System::value_type, double> &&
  requires(System& s, const System& cs, std::vector<double>& y,
           const typename Step<System>::template taken_step<double>& step,
           std::vector<solved_values_t<System>>& samples,
           std::vector<split_block<solved_values_t<System>>>& record,
           const step_record<System>& recorded) {
  { cs.sign_values(y) } -> std::same_as<void>;
  { s.split_sign_changes(step, samples, record) } -> std::same_as<bool>;
  { s.take_recorded_splits(step, recorded, samples, record) } -> std::same_as<void>;
};

template <class System>
void Step<System>::resize(size_t size_) {
  size = size_;
  for (state_type& stage_rate : k) {
    stage_rate.resize(size);
  }
}

template <class System>
size_t Step<System>::order() const {
  // In GSL, comment says "FIXME: should this be 4?"
  return 5;
}

template <class System>
void Step<System>::step(System& system,
                        solved_row& solved,
                        double time, double step_size,
                        state_type &y,
                        state_type &yerr,
                        const state_type &dydt_in,
                        state_type &dydt_out,
                        const alone_steps* alone,
                        const state_type& inputs) {
  const double h = step_size;

  // First-same-as-last: k1 is the previous step's dydt_out, so the step costs five
  // rate evaluations and one more at the state it ends at, which hands the next
  // step its own k1.
  std::copy(dydt_in.begin(), dydt_in.end(), k[0].begin());
  if constexpr (SplitsSignChanges<System>) {
    y_start = y;
  }
  std::vector<double> block_error;
  alone_samples[0].clear();
  if (alone == nullptr) {
    take_step(system, solved, time, h, y, k, y);
  } else if constexpr (StepsBlockAlone<System>) {
    alone_pass<value_type> pass{*alone, inputs};
    take_step_alone(system, solved, time, h, y, k, y, pass);
    alone_first = system.alone_block().first;
    alone_samples = std::move(pass.samples);
    block_error = std::move(pass.error);
  } else {
    util::stop("A step taken alone needs a System that names its block");
  }
  ode::derivs(system, y, dydt_out, time + h, solved.at_state);
  if constexpr (SplitsSignChanges<System>) {
    system.sign_values(sign_values[5]);
  }

  step_error(k, h, yerr);
  // A block taken alone carries the coupling's error instead.
  std::copy(block_error.begin(), block_error.end(),
            yerr.begin() + static_cast<std::ptrdiff_t>(alone_first));
}

template <class System>
std::vector<double> Step<System>::alone_ends(
    const System& system, double time, double step_size, const state_type& y,
    const state_type& dydt, const state_type& inputs,
    const std::vector<double>& slope, double step_size_min) const
  requires StepsBlockAlone<System> {
  const double h = step_size;
  const auto [first, n] = system.alone_block();
  const auto at = static_cast<std::ptrdiff_t>(first);
  const std::vector<double> to = predicted_inputs(inputs, slope, h);
  std::vector<double> own(y.begin() + at, y.begin() + at + n), next(n), stage(n),
    line(n), error(n);
  std::vector<std::vector<double>> r(6, std::vector<double>(n));
  std::copy(dydt.begin() + at, dydt.begin() + at + n, r[0].begin());
  auto rates = [&](double u, const std::vector<double>& b,
                   std::vector<double>& out) {
    for (std::size_t q = 0; q < n; ++q) {
      line[q] = inputs[q] + (to[q] - inputs[q]) * u;
    }
    system.alone_rates(time + u * h, b, line, out);
  };
  // Controlled as the solver controls its steps, in fractions of this one and
  // down to the solver's smallest step.
  OdeControl inner(1e-4 * alone_tol, alone_tol, 1.0, 0.0, step_size_min / h, 1.0,
                   1.0);
  std::vector<double> ends;
  double from = 0.0, piece = 1.0;
  for (const double stop : alone_stops) {
    while (from < stop) {
      const double end = std::min(from + piece, stop);
      for (int i = 1; i < 6; ++i) {
        stage_state(i, own, r, (end - from) * h, stage);
        rates(from + ah[i - 1] * (end - from), stage, r[i]);
      }
      step_end(own, r, (end - from) * h, next);
      step_error(r, (end - from) * h, error);
      // Compared with the piece asked for: `end - from` rounds away from it.
      const double asked = piece;
      piece = inner.adjust_step_size(n, order(), end - from, next, error, r[0]);
      if (inner.step_size_shrank()) {
        if (!(piece < asked)) {
          throw util::DomainError(
            "The System's block alone is not finite at the smallest step");
        }
        continue;
      }
      ends.push_back(end);
      from = end;
      std::swap(own, next);
      if (from < 1.0) {
        rates(from, own, r[0]);
      }
    }
  }
  return ends;
}

template <class System>
template <class S>
std::vector<S> Step<System>::predicted_inputs(const std::vector<S>& inputs,
                                              const std::vector<double>& slope,
                                              double h) {
  util::check_length(slope.size(), inputs.size());
  std::vector<S> out(inputs.size());
  for (std::size_t q = 0; q < inputs.size(); ++q) {
    out[q] = inputs[q] + h * slope[q];
    if ((util::to_passive(out[q]) < 0.0) != (util::to_passive(inputs[q]) < 0.0)) {
      out[q] = S(0.0);
    }
  }
  return out;
}

template <class System>
template <class Sys, class Row, class S>
void Step<System>::take_step_alone(Sys& sys, Row& solved, double time, double h,
                                   const std::vector<S>& y0,
                                   std::vector<std::vector<S>>& k,
                                   std::vector<S>& y_end, alone_pass<S>& alone) {
  const std::pair<std::size_t, std::size_t> block = sys.alone_block();
  const std::size_t first = block.first, n = block.second;
  const std::vector<S>& u0 = alone.inputs;
  const std::vector<double>& ends = alone.record.ends;
  util::check_length(u0.size(), n);
  if (ends.empty() || ends.back() != 1.0) {
    util::stop("A step taken alone needs inner steps that end at the step's end");
  }
  const std::vector<double> split_at(ends.begin(), ends.end() - 1);
  // The block alone over the inner steps, under inputs on the line from u0 to
  // `to`; `at_end(u, block)` sees the block where each inner step ends.
  std::vector<S> line(n);
  auto pass = [&](const std::vector<S>& to, auto&& at_end) {
    std::vector<S> own(n);
    integrate_pieces(
      first, split_at, h, y0, k,
      [&](double u, const std::vector<S>& b, std::vector<S>& out) {
        for (std::size_t q = 0; q < n; ++q) {
          line[q] = u0[q] + (to[q] - u0[q]) * u;
        }
        sys.alone_rates(time + u * h, b, line, out);
      },
      own, at_end);
    return own;
  };

  // The predictor, under the inputs the record's slope extrapolates, keeps the
  // block at each stop.
  std::array<std::vector<S>, alone_stops.size()> at;
  std::size_t next = 0;
  pass(predicted_inputs(u0, alone.record.slope, h),
       [&](double u, const std::vector<S>& b) {
         if (next < at.size() && u == alone_stops[next]) {
           at[next++] = b;
         }
       });
  if (next != at.size()) {
    util::stop("A step taken alone needs inner steps that land on every stop");
  }
  auto at_fraction = [&](double u) -> const std::vector<S>& {
    const auto stop = std::find(alone_stops.begin(), alone_stops.end(), u);
    return at.at(static_cast<std::size_t>(stop - alone_stops.begin()));
  };
  // Kept before the end is written: `y_end` may be `y0`.
  alone.samples[0].assign(y0.begin() + static_cast<std::ptrdiff_t>(first),
                          y0.begin() + static_cast<std::ptrdiff_t>(first + n));
  for (std::size_t m = 1; m < alone.samples.size(); ++m) {
    alone.samples[m] = at_fraction(taken_step<S>::sample_fractions[m]);
  }

  // The stages read the block from the predictor, and the inputs after each.
  std::vector<S> stage(y0.size());
  std::array<std::vector<S>, 5> stage_inputs;
  for (int i = 1; i < 6; ++i) {
    stage_state(i, y0, k, h, stage);
    const std::vector<S>& b = at_fraction(ah[i - 1]);
    std::copy(b.begin(), b.end(),
              stage.begin() + static_cast<std::ptrdiff_t>(first));
    ode::derivs(sys, stage, k[i], stage_time(i, time, h), solved.stages[i - 1]);
    if constexpr (SplitsSignChanges<Sys>) {
      sys.sign_values(sign_values[i - 1]);
    }
    sys.alone_inputs(stage_inputs[i - 1]);
  }

  // The corrector, to the inputs of the fourth stage, at t + h, over the same
  // inner steps; the step keeps its end.
  const std::vector<S>& u1 = stage_inputs[3];
  const std::vector<S> end = pass(u1, [](double, const std::vector<S>&) {});
  step_end(y0, k, h, y_end);
  std::copy(end.begin(), end.end(),
            y_end.begin() + static_cast<std::ptrdiff_t>(first));

  if constexpr (std::same_as<S, double>) {
    // The coupling's error: the larger of the passes' gap at the end and the
    // stages' input defect from the corrector's line, as step_end weighs stages.
    const double weight[5] = {0.0, c3, c4, 0.0, c6};
    const std::vector<S>& predicted_end = at_fraction(1.0);
    std::vector<double> with_inputs(n), on_line(n);
    alone.error.assign(n, 0.0);
    for (int i = 1; i < 6; ++i) {
      if (weight[i - 1] == 0.0) {
        continue;
      }
      for (std::size_t q = 0; q < n; ++q) {
        line[q] = u0[q] + (u1[q] - u0[q]) * ah[i - 1];
      }
      const std::vector<S>& b = at_fraction(ah[i - 1]);
      sys.alone_rates(stage_time(i, time, h), b, stage_inputs[i - 1], with_inputs);
      sys.alone_rates(stage_time(i, time, h), b, line, on_line);
      for (std::size_t q = 0; q < n; ++q) {
        alone.error[q] += h * weight[i - 1] * std::abs(with_inputs[q] - on_line[q]);
      }
    }
    for (std::size_t q = 0; q < n; ++q) {
      alone.error[q] = std::max(std::abs(end[q] - predicted_end[q]), alone.error[q]);
    }
  }
}

template <class System>
template <class Sys, class Row, class S>
void Step<System>::take_step(Sys& sys, Row& solved, double time, double h,
                             const std::vector<S>& y0,
                             std::vector<std::vector<S>>& k,
                             std::vector<S>& y_end) {
  std::vector<S> stage(y0.size());
  for (int i = 1; i < 6; ++i) {
    stage_state(i, y0, k, h, stage);
    ode::derivs(sys, stage, k[i], stage_time(i, time, h), solved.stages[i - 1]);
    if constexpr (SplitsSignChanges<Sys>) {
      sys.sign_values(sign_values[i - 1]);
    }
  }
  step_end(y0, k, h, y_end);
}

// The tableau row stage i's state combines the earlier stage rates with.
//
// The stage-0 entry is kept although no caller passes 0, and so are the stage-0
// arms of stage_time and stage_state. Dropping them and re-basing this table at
// stage 2 reads as tidying away unreachable code, and it is not: this function is
// pure and inlined, so the compiler may evaluate it above stage_state's own
// `i == 1` early return, and `rows[i - 2]` is then an out-of-bounds read of a
// stack array at i == 1. It costs nothing to keep the index at i - 1 and one
// entry in the table, and the version that removed them returned a wrong
// gradient on the second of two calls.
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
template <class S, class H>
void Step<System>::stage_state(int i, const std::vector<S>& y,
                               const std::vector<std::vector<S>>& k, const H& h,
                               std::vector<S>& out) const {
  if (i == 0) {
    std::copy(y.begin(), y.end(), out.begin());
    return;
  }
  // Stage 1 keeps its single term grouped as b21 * h * k1: h * (b21 * k1)
  // rounds differently, and the reference numbers were blessed on this one.
  if (i == 1) {
    for (size_t q = 0; q < y.size(); ++q) {
      out[q] = y[q] + b21 * h * k[0][q];
    }
    return;
  }
  const double* const b = stage_row(i);
  for (size_t q = 0; q < y.size(); ++q) {
    // Summed in ascending stage, then one h. Cash-Karp's rows are dense, so
    // this is a sum over every earlier stage rather than a term for the
    // immediate predecessor.
    S combination = b[0] * k[0][q];
    for (int m = 1; m < i; ++m) {
      combination += b[m] * k[m][q];
    }
    out[q] = y[q] + h * combination;
  }
}

template <class System>
template <class S, class H>
void Step<System>::step_end(const std::vector<S>& y,
                            const std::vector<std::vector<S>>& k, const H& h,
                            std::vector<S>& out) const {
  for (size_t q = 0; q < y.size(); ++q) {
    const S combination =
      c1 * k[0][q] + c3 * k[2][q] + c4 * k[3][q] + c6 * k[5][q];
    out[q] = y[q] + h * combination;
  }
}

template <class System>
template <class S>
void Step<System>::step_error(const std::vector<std::vector<S>>& k, double h,
                              std::vector<S>& out) const {
  for (size_t q = 0; q < out.size(); ++q) {
    out[q] = h * (ec[1] * k[0][q] + ec[3] * k[2][q] + ec[4] * k[3][q] +
                  ec[5] * k[4][q] + ec[6] * k[5][q]);
  }
}

template <class System>
template <class S, class U>
void Step<System>::dense_state(const U& u, double h, const std::vector<S>& y0,
                               const std::vector<std::vector<S>>& k,
                               const std::vector<S>& end_rate, std::size_t first,
                               std::vector<S>& out) const {
  if (first + out.size() > y0.size()) {
    util::stop("dense_state: components past the state's end");
  }
  std::array<U, 7> w;
  for (int i = 0; i < 7; ++i) {
    w[i] = (((dense_weights[3][i] * u + dense_weights[2][i]) * u +
             dense_weights[1][i]) * u + dense_weights[0][i]) * u;
  }
  // k2 carries no weight at any u, as it carries none at the end.
  for (size_t q = 0; q < out.size(); ++q) {
    const size_t i = first + q;
    const S combination =
      w[0] * k[0][i] + w[2] * k[2][i] + w[3] * k[3][i] + w[4] * k[4][i] +
      w[5] * k[5][i] + w[6] * end_rate[i];
    out[q] = y0[i] + h * combination;
  }
}

template <class System>
template <class S>
template <class U>
void Step<System>::taken_step<S>::sample_at(
    const U& u, const std::array<std::vector<S>, 5>& samples,
    std::vector<S>& out) const {
  std::array<U, 5> w;
  for (std::size_t m = 0; m < w.size(); ++m) {
    w[m] = U(1.0);
    for (std::size_t n = 0; n < w.size(); ++n) {
      if (n != m) {
        w[m] *= (u - sample_fractions[n]) /
                (sample_fractions[m] - sample_fractions[n]);
      }
    }
  }
  out.resize(samples[0].size());
  for (const std::vector<S>& sample : samples) {
    util::check_length(sample.size(), out.size());
  }
  for (std::size_t q = 0; q < out.size(); ++q) {
    out[q] = w[0] * samples[0][q] + w[1] * samples[1][q] + w[2] * samples[2][q] +
             w[3] * samples[3][q] + w[4] * samples[4][q];
  }
}

template <class System>
template <class S, class U, class Rates, class AtEnd>
void Step<System>::integrate_pieces(std::size_t first,
                                    const std::vector<U>& split_at, double h,
                                    const std::vector<S>& y0,
                                    const std::vector<std::vector<S>>& k,
                                    Rates&& rates, std::vector<S>& own,
                                    AtEnd&& at_end) const {
  const std::size_t width = own.size();
  const auto at = static_cast<std::ptrdiff_t>(first);
  const auto end = at + static_cast<std::ptrdiff_t>(width);
  std::vector<std::vector<S>> piece_rates(6, std::vector<S>(width));
  std::vector<S> stage(width);
  std::copy(y0.begin() + at, y0.begin() + end, own.begin());
  U from = 0.0;
  for (std::size_t c = 0; c <= split_at.size(); ++c) {
    const U to = c < split_at.size() ? split_at[c] : U(1.0);
    if (!(util::to_passive(to) > util::to_passive(from))) {
      continue;
    }
    const U piece = (to - from) * h;
    // At the step's start the block's rates are the step's own first ones.
    if (util::to_passive(from) == 0.0) {
      std::copy(k[0].begin() + at, k[0].begin() + end, piece_rates[0].begin());
    } else {
      rates(from, own, piece_rates[0]);
    }
    for (int i = 1; i < 6; ++i) {
      stage_state(i, own, piece_rates, piece, stage);
      rates(from + ah[i - 1] * (to - from), stage, piece_rates[i]);
    }
    step_end(own, piece_rates, piece, own);
    at_end(to, own);
    from = to;
  }
}

template <class System>
template <class S>
template <class Values, class ValueAt>
std::vector<sign_change<Values>>
Step<System>::taken_step<S>::sign_changes(std::size_t block,
                                          ValueAt&& value_at) const
  requires std::same_as<S, double> {
  const std::array<std::vector<double>, 6>& stages = *sign_values;
  const double v0 = (*sign_values_in)[block], v1 = stages[5][block];
  const bool negative = v0 < 0.0;
  auto other = [&](double v) { return (v < 0.0) != negative; };
  const double start_sign = negative ? -1.0 : 1.0;
  std::vector<sign_change<Values>> changes;
  // A reading within this share of the readings' spread from zero is searched
  // around for a pair.
  const double near_zero = 0.02;

  // The zero between fractions a and b, whose values va and vb differ in sign:
  // regula falsi, halving a retained end's value (Illinois), to 1e-10 or 64 steps.
  auto locate = [&](double a, double va, double b, double vb) {
    sign_change<Values>& change = changes.emplace_back();
    const double settled = 1e-10;  // in fractions of the step
    double u = std::numeric_limits<double>::quiet_NaN();
    int side = 0;
    for (int iteration = 0; iteration < 64; ++iteration) {
      const double next = (a * vb - b * va) / (vb - va);
      const double v = value_at(next, &change.solved);
      if (v == 0.0 || std::abs(next - u) < settled || iteration == 63) {
        // A central difference: the iterates can stay at one end of the bracket,
        // where a secant misses a curved sign value.
        const double d = 1e-5;
        change.u = next;
        change.slope =
          (value_at(next + d, nullptr) - value_at(next - d, nullptr)) / (2 * d);
        return;
      }
      u = next;
      if ((v < 0.0) == (vb < 0.0)) {
        b = u;
        vb = v;
        if (side == -1) {
          va /= 2;
        }
        side = -1;
      } else {
        a = u;
        va = v;
        if (side == 1) {
          vb /= 2;
        }
        side = 1;
      }
    }
  };

  // Where on (a, b) the sign value lies furthest toward the other sign: a golden
  // section of at most four dense-output values, stopping at a value of that sign.
  auto search = [&](double a, double b) -> std::pair<double, double> {
    const double g = 0.5 * (3.0 - std::sqrt(5.0));
    double x1 = a + g * (b - a), x2 = b - g * (b - a);
    double f1 = start_sign * value_at(x1, nullptr);
    if (f1 < 0.0) {
      return {x1, start_sign * f1};
    }
    double f2 = start_sign * value_at(x2, nullptr);
    for (int more = 0; more < 2 && f2 >= 0.0; ++more) {
      if (f1 < f2) {
        b = x2;
        x2 = x1;
        f2 = f1;
        x1 = a + g * (b - a);
        f1 = start_sign * value_at(x1, nullptr);
        if (f1 < 0.0) {
          return {x1, start_sign * f1};
        }
      } else {
        a = x1;
        x1 = x2;
        f1 = f2;
        x2 = b - g * (b - a);
        f2 = start_sign * value_at(x2, nullptr);
      }
    }
    return f1 < f2 ? std::pair{x1, start_sign * f1}
                   : std::pair{x2, start_sign * f2};
  };

  if (other(v1)) {
    locate(0.0, v0, 1.0, v1);
    return changes;
  }
  // A pair: the end or inner-stage reading furthest toward the other sign, if it
  // has that sign or lies near zero; a stage is read again on the dense output.
  const double at[6] = {0.0, ah[0], ah[1], ah[2], ah[4], 1.0};
  const double read[6] = {v0, stages[0][block], stages[1][block],
                          stages[2][block], stages[4][block], v1};
  int m = 0;
  double low = v0, high = v0;
  for (int k = 1; k < 6; ++k) {
    if (start_sign * read[k] < start_sign * read[m]) {
      m = k;
    }
    low = std::min(low, read[k]);
    high = std::max(high, read[k]);
  }
  if (start_sign * read[m] > near_zero * (high - low)) {
    return changes;
  }
  double um = at[m];
  double vm = m == 0 || m == 5 ? read[m] : value_at(um, nullptr);
  for (int k : {m - 1, m + 1}) {
    if (!other(vm) && k >= 0 && k <= 5) {
      std::tie(um, vm) =
        search(std::min(at[k], at[m]), std::max(at[k], at[m]));
    }
  }
  if (other(vm)) {
    locate(0.0, v0, um, vm);
    locate(um, vm, 1.0, v1);
  }
  return changes;
}

// lambda_in[m] = (d y_end / d y)^T lambda_out[m] for the one step step() takes
// from y, and the parameter rows alongside it.
//
// ONE recording spans the whole step: its six rate evaluations and the
// combination closing them. What the sweep transposes is therefore the
// arithmetic the stepper performs, where a recording per stage left the tableau
// to be transposed by hand beside the stepper and held consistent with it by
// discipline. The stage states are intermediates of the recording rather than a
// double rebuild ahead of it, so the step costs six model evaluations and not
// thirteen.
//
// The recording is derivs(), which is what the forward pass calls, so no System
// writes a transpose of its own; and the parameters ride in the same recording,
// so a stage the parameters reach carries their rows too.
template <class System>
void Step<System>::step_adjoint(active_system<System>& active,
                                const solved_values& first, double first_time,
                                const solved_row& solved,
                                double time, double step_size,
                                const state_type &y, const adjoint_rows& lambda_out,
                                adjoint_rows& lambda_in,
                                adjoint_rows& parameter_adjoint,
                                const alone_steps& alone) {
  using scalar = active_scalar<double>;
  const double h = step_size;
  if (lambda_out.empty()) {
    util::stop("step_adjoint: needs at least one seed");
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
    // k1 repeats the evaluation the row below recorded at this state.
    ode::derivs(sys, y0, rate[0], first_time, first);
    std::array<std::vector<scalar>, 5> samples;
    std::size_t block_first = 0;
    if (alone.slope.empty()) {
      take_step(sys, solved, time, h, y0, rate, y_end);
    } else if constexpr (StepsBlockAlone<std::remove_cvref_t<decltype(sys)>>) {
      // The block's inputs at the start, off the evaluation that just ran there.
      std::vector<scalar> inputs;
      sys.alone_inputs(inputs);
      alone_pass<scalar> pass{alone, inputs};
      take_step_alone(sys, solved, time, h, y0, rate, y_end, pass);
      samples = std::move(pass.samples);
      block_first = sys.alone_block().first;
    } else {
      util::stop("A step taken alone needs a System that names its block");
    }
    recorded_rates += 6;
    if constexpr (SplitsSignChanges<System>) {
      if (!solved.split_blocks.empty()) {
        // The dense output reads the rates at the end before the split, and the
        // System, lifted to the active scalar, splits as it recorded.
        std::vector<scalar> end_rate(size);
        ode::derivs(sys, y_end, end_rate, time + h, solved.at_state_before_split);
        ++recorded_rates;
        sys.split_as_recorded(
          taken_step<scalar>{*this, time, h, y0, rate, end_rate, y_end, nullptr,
                             nullptr, samples[0].empty() ? nullptr : &samples,
                             block_first},
          solved.samples, solved.split_blocks);
      }
    }
  };

  ode::state_and_parameter_adjoints(active, y, lambda_out, whole_step, lambda_in,
                                    parameter_adjoint);
}

// RKCK coefficients, from GSL
template <class System>
const double Step<System>::ah[] = {
  1.0 / 5.0, 0.3, 3.0 / 5.0, 1.0, 7.0 / 8.0 };

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

// The C1 quartic whose order-5 error at u = 1/2 is least; at u = 1 its weights are
// the step's own. Without the end's rate the six stages admit no order-4 output.
template <class System>
const double Step<System>::dense_weights[4][7] = {
  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {-156473.0 / 57792.0, 0.0, 1159825.0 / 332304.0, 14725.0 / 60544.0,
   5301.0 / 38528.0, -202836.0 / 76153.0, 3.0 / 2.0},
  {729889.0 / 260064.0, 0.0, -8030425.0 / 1495368.0, 290425.0 / 817344.0,
   -5301.0 / 19264.0, 493736.0 / 76153.0, -4.0},
  {-24797.0 / 24768.0, 0.0, 2275475.0 / 996912.0, -19225.0 / 49536.0,
   5301.0 / 38528.0, -3492.0 / 989.0, 5.0 / 2.0}};

}
}

#endif
