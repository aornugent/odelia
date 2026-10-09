// -*-c++-*-
#ifndef ODELIA_ODE_SUBSTEPS_HPP_
#define ODELIA_ODE_SUBSTEPS_HPP_

#include <odelia/ode_step.hpp>

// Substeps: a block of the state integrated over parts of one step. The split
// integrates a node's block in substeps meeting at its sign changes; a System's
// subsystem is integrated in substeps of its own under inputs on a line.

namespace odelia {
namespace ode {

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
  const std::vector<double> to = extrapolated_inputs(u0, input_slope, h);
  auto rates = rate_on_line(system, time, h, u0, to);
  const auto last = at + static_cast<std::ptrdiff_t>(n);
  std::vector<double> s(y.begin() + at, y.begin() + last), next(n), error(n);
  std::vector<std::vector<double>> r(6, std::vector<double>(n));
  std::copy(dydt.begin() + at, dydt.begin() + last, r[0].begin());
  // Controlled as the solver controls its steps, in fractions of this one and
  // down to the solver's smallest step.
  OdeControl control(1e-4 * substep_tol, substep_tol, 1.0, 0.0, step_size_min / h,
                     1.0, 1.0);
  subsystem_substeps chosen{std::move(input_slope), {}};
  double from = 0.0, substep = 1.0;
  for (const double fraction : stage_and_sample_fractions) {
    while (from < fraction) {
      const double end = std::min(from + substep, fraction);
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
template <class Sys, class S, class AtEnd>
std::vector<S> Step<System>::integrate_subsystem(
    const Sys& sys, double time, double h, const std::vector<S>& y0,
    const std::vector<std::vector<S>>& k, const std::vector<double>& ends,
    const std::vector<S>& from, const std::vector<S>& to, AtEnd&& at_end) const {
  const state_range subsystem = sys.subsystem();
  const std::vector<double> split_at(ends.begin(), ends.end() - 1);
  std::vector<S> s(subsystem.size);
  integrate_substeps(subsystem.first, split_at, h, y0, k,
                     rate_on_line(sys, time, h, from, to), s, at_end);
  return s;
}

template <class System>
template <class Sys, class Row, class S>
std::vector<double> Step<System>::take_substepped_step(
    Sys& sys, Row& solved, double time, double h, const std::vector<S>& y0,
    std::vector<std::vector<S>>& k, std::vector<S>& y_end,
    const subsystem_substeps& substeps, const std::vector<S>& u0,
    subsystem_samples<S>& samples) {
  const state_range subsystem = sys.subsystem();
  const auto first = static_cast<std::ptrdiff_t>(subsystem.first);
  const auto last = first + static_cast<std::ptrdiff_t>(subsystem.size);
  util::check_length(u0.size(), subsystem.size);
  if (substeps.ends.empty() || substeps.ends.back() != 1.0) {
    util::stop("Substeps must end at the step's end");
  }

  // Predictor: the subsystem over the substeps, its inputs extrapolated from u0
  // along the recorded slope. Kept at each stage and sample fraction.
  std::array<std::vector<S>, stage_and_sample_fractions.size()> predicted;
  std::size_t next = 0;
  integrate_subsystem(sys, time, h, y0, k, substeps.ends, u0,
                      extrapolated_inputs(u0, substeps.input_slope, h),
                      [&](double u, const std::vector<S>& s) {
                        if (next < predicted.size() &&
                            u == stage_and_sample_fractions[next]) {
                          predicted[next++] = s;
                        }
                      });
  if (next != predicted.size()) {
    util::stop("Substeps must land on every stage and sample fraction");
  }
  auto predicted_at = [&](double u) -> const std::vector<S>& {
    const auto at = std::find(stage_and_sample_fractions.begin(),
                              stage_and_sample_fractions.end(), u);
    return predicted.at(
      static_cast<std::size_t>(at - stage_and_sample_fractions.begin()));
  };
  // Kept before the end is written: `y_end` may be `y0`.
  samples.first = subsystem.first;
  samples.at[0].assign(y0.begin() + first, y0.begin() + last);
  for (std::size_t m = 1; m < samples.at.size(); ++m) {
    samples.at[m] = predicted_at(taken_step<S>::sample_fractions[m]);
  }

  // Stages: the rest from the tableau, the subsystem from the predictor. Each
  // evaluation leaves that stage's inputs on the System.
  std::vector<S> stage(y0.size());
  std::array<std::vector<S>, 5> stage_inputs;
  for (int i = 1; i < 6; ++i) {
    stage_state(i, y0, k, h, stage);
    const std::vector<S>& s = predicted_at(ah[i - 1]);
    std::copy(s.begin(), s.end(), stage.begin() + first);
    ode::derivs(sys, stage, k[i], stage_time(i, time, h), solved.stages[i - 1]);
    if constexpr (SplitsSignChanges<Sys>) {
      sys.sign_values(sign_values[i - 1]);
    }
    sys.subsystem_inputs(stage_inputs[i - 1]);
  }

  // Corrector: the subsystem again over the same substeps, its inputs on the
  // line from u0 to those at the stage at t + h. Its end is the step's.
  constexpr auto stage_at_end = static_cast<std::size_t>(
    std::find(std::begin(ah), std::end(ah), 1.0) - std::begin(ah));
  const std::vector<S>& inputs_at_end = stage_inputs[stage_at_end];
  const std::vector<S> end =
    integrate_subsystem(sys, time, h, y0, k, substeps.ends, u0, inputs_at_end,
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
    auto corrector_line = rate_on_line(sys, time, h, u0, inputs_at_end);
    std::vector<double> stage_rates(subsystem.size), line_rates(subsystem.size);
    error.assign(subsystem.size, 0.0);
    for (int i = 1; i < 6; ++i) {
      if (weight[i - 1] == 0.0) {
        continue;
      }
      const std::vector<S>& s = predicted_at(ah[i - 1]);
      sys.subsystem_rates(stage_time(i, time, h), s, stage_inputs[i - 1],
                          stage_rates);
      corrector_line(ah[i - 1], s, line_rates);
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
template <class S, class U, class Rates, class AtEnd>
void Step<System>::integrate_substeps(std::size_t first,
                                      const std::vector<U>& split_at, double h,
                                      const std::vector<S>& y0,
                                      const std::vector<std::vector<S>>& k,
                                      Rates&& rates, std::vector<S>& own,
                                      AtEnd&& at_end) const {
  const std::size_t width = own.size();
  const auto at = static_cast<std::ptrdiff_t>(first);
  const auto end = at + static_cast<std::ptrdiff_t>(width);
  std::vector<std::vector<S>> substep_rates(6, std::vector<S>(width));
  std::copy(y0.begin() + at, y0.begin() + end, own.begin());
  U from = 0.0;
  for (std::size_t c = 0; c <= split_at.size(); ++c) {
    const U to = c < split_at.size() ? split_at[c] : U(1.0);
    if (!(util::to_passive(to) > util::to_passive(from))) {
      continue;
    }
    // At the step's start the block's rates are the step's own first ones.
    if (util::to_passive(from) == 0.0) {
      std::copy(k[0].begin() + at, k[0].begin() + end, substep_rates[0].begin());
    } else {
      rates(from, own, substep_rates[0]);
    }
    take_substep(own, substep_rates, from, to, h, rates, own);
    at_end(to, own);
    from = to;
  }
}

template <class System>
template <class S, class U, class Rates>
void Step<System>::take_substep(const std::vector<S>& own,
                                std::vector<std::vector<S>>& r, const U& from,
                                const U& to, double h, Rates&& rates,
                                std::vector<S>& out) const {
  const U substep = (to - from) * h;
  std::vector<S> stage(own.size());
  for (int i = 1; i < 6; ++i) {
    stage_state(i, own, r, substep, stage);
    rates(from + ah[i - 1] * (to - from), stage, r[i]);
  }
  step_end(own, r, substep, out);
}

}
}

#endif
