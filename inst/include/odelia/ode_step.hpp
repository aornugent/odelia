// -*-c++-*-
#ifndef ODELIA_ODE_STEP_HPP_
#define ODELIA_ODE_STEP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <cstddef>
#include <XAD/XAD.hpp>
#include <odelia/adjoint.hpp>
#include <odelia/ode_interface.hpp>

namespace odelia {
namespace ode {

// By part, over the steps kept since the last reset: the steps split and the steps
// searched beside a reading near zero; then the slowest crossing at a cut.
struct split_record {
  std::vector<std::size_t> split;
  std::vector<std::size_t> searched;
  // |d(sign value)/dt| at that cut, when it fell and in which part.
  double least_rate = std::numeric_limits<double>::infinity();
  double least_rate_time = std::numeric_limits<double>::quiet_NaN();
  std::size_t least_rate_part = 0;
  std::size_t total() const {
    return std::accumulate(split.begin(), split.end(), std::size_t{0});
  }
};

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
  void step(System& system, solved_row& solved,
            double time, double step_size,
	    state_type &y,
	    state_type &yerr,
	    const state_type &dydt_in,
	    state_type &dydt_out);

  // Each part whose sign value changes sign in the step just taken, integrated
  // again in pieces between its sign changes; then the corrected end rated again.
  void split(System& system, solved_values& at_state, double time,
             double step_size, const std::vector<double>& sign_values_in,
             state_type& y, state_type& dydt_out)
    requires SplitsSignChanges<System>;

  // The state at fraction u of the step just taken, on Cash-Karp's fourth-order
  // continuous extension through its stage rates and the end's rate.
  void dense_state(double u, double step_size, const state_type& dydt_out,
                   state_type& out) const;

  // Each part's sign value where the step just taken ended, which the next step
  // starts from.
  const std::vector<double>& end_sign_values() const { return sign_values[5]; }

  // The step transposed, for as many seeds as are handed in: one recording of the
  // whole step, swept once per seed. The active System is the walk's, held across
  // every step of one width. A caller wanting one row passes a batch of
  // one; there is no separate entry point for that, because a second signature
  // over the same recording is a second place for the seam between the state and
  // the parameter halves to be got wrong.
  //
  // `first` is what the evaluation at the step's start state solved for, and
  // `first_time` when it ran: the row below's `at_state`.
  void step_adjoint(active_system<System>& active,
                    const solved_values& first, double first_time,
                    const solved_row& solved,
                    double time, double step_size,
                    const state_type &y, const adjoint_rows& lambda_out,
                    adjoint_rows& lambda_in, adjoint_rows& parameter_adjoint);

  // Rate evaluations the sweeps since the last clear have recorded, counted where
  // they are recorded rather than added up as a total the loop could disagree
  // with. Six a step, whatever the seed count, because the step is recorded once
  // and swept per seed. A term entering once a step where it belongs once a stage
  // divides this by six, and no gradient check can see that, because a tangent and
  // a sweep apply the same multiplier.
  std::size_t recorded_rates = 0;
  // What the splits did on the steps kept since the solver's last reset, pinned
  // or not.
  split_record splits;

  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // The tableau, written once and used at whatever scalar the caller holds its
  // rates in: the forward step and the recording its transpose is taken from
  // both step through these, so the two cannot come apart.
  //
  // Y_i for stage i, into `out`: y at stage 0, and y plus the combination of the
  // earlier stage rates above that. Callers pass 1..5; see stage_row for why the
  // stage-0 arms stay. Both run over y's length, so a part's integration reuses
  // them.
  template <class S>
  void stage_state(int i, const std::vector<S>& y,
                   const std::vector<std::vector<S>>& k, double h,
                   std::vector<S>& out) const;
  // And the state the step ends at, y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6).
  // k2 and k5 reach it only through the later stages. `out` may be `y`.
  template <class S>
  void step_end(const std::vector<S>& y, const std::vector<std::vector<S>>& k,
                double h, std::vector<S>& out) const;
  double stage_time(int i, double time, double h) const;
  const double* stage_row(int i) const;

  size_t size;
  std::vector<state_type> k{6};
  state_type ytmp;
  // The state the step just taken started from, which the dense output reads.
  state_type y_start;
  // Each part's sign value at the five stages, then at the end, of that step.
  std::array<std::vector<double>, 6> sign_values;

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
void Step<System>::step(System& system,
                        solved_row& solved,
                        double time, double step_size,
                        state_type &y,
                        state_type &yerr,
                        const state_type &dydt_in,
                        state_type &dydt_out) {
  const double h = step_size;

  // First-same-as-last: k1 is the previous step's dydt_out, so the step costs five
  // rate evaluations and one more at the state it ends at, which hands the next
  // step its own k1.
  y_start = y;
  std::copy(dydt_in.begin(), dydt_in.end(), k[0].begin());
  for (int i = 1; i < 6; ++i) {
    stage_state(i, y, k, h, ytmp);
    ode::derivs(system, ytmp, k[i], stage_time(i, time, h), solved.stages[i - 1]);
    if constexpr (SplitsSignChanges<System>) {
      system.sign_values(sign_values[i - 1]);
    }
  }

  step_end(y, k, h, y);
  ode::derivs(system, y, dydt_out, time + h, solved.at_state);
  if constexpr (SplitsSignChanges<System>) {
    system.sign_values(sign_values[5]);
  }

  // Difference between 4th and 5th order, for error calculations
  for (size_t q = 0; q < size; ++q) {
    yerr[q] = h * (ec[1] * k[0][q] + ec[3] * k[2][q] + ec[4] * k[3][q] +
                   ec[5] * k[4][q] + ec[6] * k[5][q]);
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
template <class S>
void Step<System>::stage_state(int i, const std::vector<S>& y,
                               const std::vector<std::vector<S>>& k, double h,
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
template <class S>
void Step<System>::step_end(const std::vector<S>& y,
                            const std::vector<std::vector<S>>& k, double h,
                            std::vector<S>& out) const {
  for (size_t q = 0; q < y.size(); ++q) {
    const S combination =
      c1 * k[0][q] + c3 * k[2][q] + c4 * k[3][q] + c6 * k[5][q];
    out[q] = y[q] + h * combination;
  }
}

template <class System>
void Step<System>::dense_state(double u, double step_size,
                               const state_type& dydt_out,
                               state_type& out) const {
  double w[7];
  for (int i = 0; i < 7; ++i) {
    w[i] = (((dense_weights[3][i] * u + dense_weights[2][i]) * u +
             dense_weights[1][i]) * u + dense_weights[0][i]) * u;
  }
  // k2 carries no weight at any u, as it carries none at the end.
  for (size_t q = 0; q < size; ++q) {
    const value_type combination =
      w[0] * k[0][q] + w[2] * k[2][q] + w[3] * k[3][q] + w[4] * k[4][q] +
      w[5] * k[5][q] + w[6] * dydt_out[q];
    out[q] = y_start[q] + step_size * combination;
  }
}

// Split once where the ends' sign values differ, and twice where a stage inside,
// or a search beside a reading near zero, finds the other sign on the dense output.
template <class System>
void Step<System>::split(System& system, solved_values& at_state, double time,
                         double step_size,
                         const std::vector<double>& sign_values_in,
                         state_type& y, state_type& dydt_out)
  requires SplitsSignChanges<System> {
  const std::size_t parts = sign_values_in.size();
  if (parts == 0) {
    return;
  }
  for (const std::vector<double>& values : sign_values) {
    util::check_length(values.size(), parts);
  }
  if (splits.split.size() < parts) {
    splits.split.resize(parts);
    splits.searched.resize(parts);
  }
  const std::size_t width = system.part_width();
  if (parts * width > size) {
    util::stop("split: " + util::to_string(static_cast<int>(parts)) +
               " parts of " + util::to_string(static_cast<int>(width)) +
               " components exceed a state of " +
               util::to_string(static_cast<int>(size)));
  }
  const double h = step_size;
  // A reading this near zero, as a share of the readings' spread over the step,
  // has the gaps beside it searched for a pair.
  const double near_zero = 0.02;
  // Each part rating moves the System off the step's end, so the end is rated
  // again whenever one ran, whether or not a part was split.
  bool rated = false;
  std::vector<double> rates(width);
  // Part p's sign value at fraction u, its rates into `out`; its own components
  // are `own` where given, else the dense output's.
  auto rate = [&](std::size_t p, double u, const std::vector<double>* own,
                  std::vector<double>& out) -> double {
    dense_state(u, h, dydt_out, ytmp);
    if (own != nullptr) {
      std::copy(own->begin(), own->end(),
                ytmp.begin() + static_cast<std::ptrdiff_t>(p * width));
    }
    rated = true;
    return system.part_rates(p, ytmp, time + u * h, out);
  };
  // The sign change between fractions a and b, where the sign value is va and vb
  // of opposite signs: regula falsi, halving a retained end's value (Illinois).
  auto locate = [&](std::size_t p, double a, double va, double b,
                    double vb) -> double {
    const double settled = 1e-10;  // in fractions of the step
    double u = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 65> taken_u, taken_v;
    taken_u[0] = b;
    taken_v[0] = vb;
    int taken = 1;
    int side = 0;
    for (int iteration = 0; iteration < 64; ++iteration) {
      const double next = (a * vb - b * va) / (vb - va);
      const double v = rate(p, next, nullptr, rates);
      if (v == 0.0 || std::abs(next - u) < settled) {
        // The slope there, against the latest value at least 1e-6 of the step
        // away: nearer, the difference is roundoff.
        int far = taken - 1;
        while (far > 0 && std::abs(taken_u[far] - next) < 1e-6) {
          --far;
        }
        const double crossing =
          std::abs((v - taken_v[far]) / (next - taken_u[far])) / h;
        if (crossing < splits.least_rate) {
          splits.least_rate = crossing;
          splits.least_rate_time = time + next * h;
          splits.least_rate_part = p;
        }
        return next;
      }
      u = next;
      taken_u[taken] = next;
      taken_v[taken] = v;
      ++taken;
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
    return u;
  };

  // Where on (a, b) the sign value leans furthest to the other sign: a golden
  // section of at most four dense-output values, which stops at the first such.
  auto search = [&](std::size_t p, double a, double b,
                    bool negative) -> std::pair<double, double> {
    const double lean = negative ? -1.0 : 1.0;
    const double g = 0.5 * (3.0 - std::sqrt(5.0));
    double x1 = a + g * (b - a), x2 = b - g * (b - a);
    double f1 = lean * rate(p, x1, nullptr, rates);
    if (f1 < 0.0) {
      return {x1, lean * f1};
    }
    double f2 = lean * rate(p, x2, nullptr, rates);
    for (int more = 0; more < 2 && f2 >= 0.0; ++more) {
      if (f1 < f2) {
        b = x2;
        x2 = x1;
        f2 = f1;
        x1 = a + g * (b - a);
        f1 = lean * rate(p, x1, nullptr, rates);
        if (f1 < 0.0) {
          return {x1, lean * f1};
        }
      } else {
        a = x1;
        x1 = x2;
        f1 = f2;
        x2 = b - g * (b - a);
        f2 = lean * rate(p, x2, nullptr, rates);
      }
    }
    return f1 < f2 ? std::pair{x1, lean * f1} : std::pair{x2, lean * f2};
  };

  std::vector<std::vector<double>> piece_rates(6, std::vector<double>(width));
  std::vector<double> own(width), stage(width);
  for (std::size_t p = 0; p < parts; ++p) {
    const auto first = static_cast<std::ptrdiff_t>(p * width);
    const double v0 = sign_values_in[p], v1 = sign_values[5][p];
    const bool negative = v0 < 0.0;
    double cuts[2];
    int n_cuts = 0;
    if ((v1 < 0.0) != negative) {
      cuts[n_cuts++] = locate(p, 0.0, v0, 1.0, v1);
    } else {
      int deepest = 0;
      for (int i = 1; i < 6; ++i) {
        const double v = sign_values[i - 1][p];
        if ((v < 0.0) != negative && ah[i - 1] < 1.0 &&
            (deepest == 0 ||
             std::abs(v) > std::abs(sign_values[deepest - 1][p]))) {
          deepest = i;
        }
      }
      if (deepest > 0) {
        const double um = ah[deepest - 1];
        const double vm = rate(p, um, nullptr, rates);
        if ((vm < 0.0) != negative) {
          cuts[n_cuts++] = locate(p, 0.0, v0, um, vm);
          cuts[n_cuts++] = locate(p, um, vm, 1.0, v1);
        }
      } else {
        // No reading holds the other sign. A pair may sit beside the one nearest
        // zero if it is within near_zero of their spread, so its gaps are searched.
        const double at[6] = {0.0, ah[0], ah[1], ah[2], ah[4], 1.0};
        const double read[6] = {v0, sign_values[0][p], sign_values[1][p],
                                sign_values[2][p], sign_values[4][p], v1};
        int nearest = 0;
        double low = v0, high = v0;
        for (int k = 1; k < 6; ++k) {
          if (std::abs(read[k]) < std::abs(read[nearest])) {
            nearest = k;
          }
          low = std::min(low, read[k]);
          high = std::max(high, read[k]);
        }
        if (std::abs(read[nearest]) <= near_zero * (high - low)) {
          ++splits.searched[p];
          for (int k : {nearest - 1, nearest + 1}) {
            if (k < 0 || k > 5 || n_cuts > 0) {
              continue;
            }
            const auto [um, vm] = search(p, std::min(at[k], at[nearest]),
                                         std::max(at[k], at[nearest]), negative);
            if ((vm < 0.0) != negative) {
              cuts[n_cuts++] = locate(p, 0.0, v0, um, vm);
              cuts[n_cuts++] = locate(p, um, vm, 1.0, v1);
            }
          }
        }
      }
    }
    if (n_cuts == 0) {
      continue;
    }
    ++splits.split[p];

    std::copy(y_start.begin() + first,
              y_start.begin() + first + static_cast<std::ptrdiff_t>(width),
              own.begin());
    double from = 0.0;
    for (int c = 0; c <= n_cuts; ++c) {
      const double to = c < n_cuts ? cuts[c] : 1.0;
      if (!(to > from)) {
        continue;
      }
      const double piece = (to - from) * h;
      // At the step's start the part's rates are the step's own first ones.
      if (from == 0.0) {
        std::copy(k[0].begin() + first,
                  k[0].begin() + first + static_cast<std::ptrdiff_t>(width),
                  piece_rates[0].begin());
      } else {
        rate(p, from, &own, piece_rates[0]);
      }
      for (int i = 1; i < 6; ++i) {
        stage_state(i, own, piece_rates, piece, stage);
        rate(p, from + ah[i - 1] * (to - from), &stage, piece_rates[i]);
      }
      step_end(own, piece_rates, piece, own);
      from = to;
    }
    std::copy(own.begin(), own.end(), y.begin() + first);
  }

  if (rated) {
    ode::derivs(system, y, dydt_out, time + h, at_state);
    system.sign_values(sign_values[5]);
  }
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
                                adjoint_rows& parameter_adjoint) {
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
    std::vector<scalar> stage(size);
    // k1 repeats the evaluation the row below recorded at this state.
    ode::derivs(sys, y0, rate[0], first_time, first);
    ++recorded_rates;
    for (int i = 1; i < 6; ++i) {
      stage_state(i, y0, rate, h, stage);
      ode::derivs(sys, stage, rate[i], stage_time(i, time, h), solved.stages[i - 1]);
      ++recorded_rates;
    }
    step_end(y0, rate, h, y_end);
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
