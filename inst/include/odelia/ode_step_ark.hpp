// -*-c++-*-
#ifndef ODELIA_ODE_STEP_ARK_HPP_
#define ODELIA_ODE_STEP_ARK_HPP_

// ARK4(3)6L[2]SA (Kennedy & Carpenter 2003): the explicit tableau steps the whole
// state, the diagonally implicit one the System's stiff block, by damped Newton.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <XAD/XAD.hpp>
#include <odelia/adjoint.hpp>
#include <odelia/implicit_node.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_linalg.hpp>
#include <odelia/ode_step.hpp>
#include <odelia/ode_util.hpp>

namespace odelia {
namespace ode {

// The stiff block alone, its inputs from the rest of the state linear in time
// between their values at an attempt's two ends: what estimates the block's error.
template <class System>
struct block_alone {
  using value_type = double;

  const System* system;
  double start, step_size;
  const std::vector<double>* inputs_start;
  const std::vector<double>* inputs_end;
  std::vector<double> y, inputs, rates;
  double time = 0.0;

  size_t ode_size() const { return y.size(); }
  double ode_time() const { return time; }

  template <class It> It set_ode_state(It it, double time_) {
    for (double& x : y) {
      x = *it++;
    }
    time = time_;
    for (size_t q = 0; q < y.size(); ++q) {
      inputs[q] = (*inputs_start)[q] +
                  ((*inputs_end)[q] - (*inputs_start)[q]) * (time - start) /
                      step_size;
    }
    system->stiff_alone(time, y, inputs, rates);
    return it;
  }
  template <class It> It ode_state(It it) const {
    for (double x : y) {
      *it++ = x;
    }
    return it;
  }
  template <class It> It ode_rates(It it) const {
    for (double x : rates) {
      *it++ = x;
    }
    return it;
  }
};

template <class System>
class ArkStep {
public:
  using value_type = typename System::value_type;
  using state_type = std::vector<value_type>;

  using solved_values = solved_values_t<System>;
  using solved_row = ode::solved_row<solved_values>;

  // Whether the System names a stiff block for the implicit tableau to step.
  static constexpr bool supported = HasStiffBlock<System>;

  void resize(size_t size_);
  size_t order() const { return 4; }

  // `solved` is the row this step creates, as for Step: its five stages' slots
  // and the evaluation at the state it ends at.
  void step(System& system, solved_row& solved, double time, double step_size,
            state_type& y, state_type& yerr, const state_type& dydt_in,
            state_type& dydt_out);

  // The block's estimate in place of the embedded one: the block alone over the
  // attempt, Cash-Karp at block_tol, inputs linear from start_inputs' to the end's.
  void start_inputs(const System& system);
  void block_error(const System& system, double time, double step_size,
                   const state_type& y_start, const state_type& y_end,
                   state_type& yerr);

  // The step transposed, as Step::step_adjoint. Each implicit stage enters the
  // recording by the implicit function theorem at the stage it converged to.
  void step_adjoint(active_system<System>& active,
                    const solved_values& first, double first_time,
                    const solved_row& solved, double time, double step_size,
                    const state_type& y, const adjoint_rows& lambda_out,
                    adjoint_rows& lambda_in, adjoint_rows& parameter_adjoint);

  // Rate evaluations the sweeps since the last clear have recorded: six a step,
  // as Cash-Karp's.
  std::size_t recorded_rates = 0;

  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // The stages and the end, at the caller's scalar, so the forward step and its
  // transpose's recording cannot part; k[0] holds y0's rates; `y_end` may be `y0`.
  template <class Sys, class Row, class S>
  void take_step(Sys& sys, Row& solved, double time, double h,
                 const std::vector<S>& y0, std::vector<std::vector<S>>& k,
                 std::vector<std::vector<S>>& stiff,
                 std::vector<S>& y_end) const;
  // Stage i's explicit combination: a one-term row grouped as (a h) k, the rest
  // summed in ascending stage, then one h.
  template <class S>
  void explicit_stage(int i, const std::vector<S>& y0,
                      const std::vector<std::vector<S>>& k, double h,
                      std::vector<S>& out) const;
  template <class S>
  void step_end(const std::vector<S>& y0, const std::vector<std::vector<S>>& k,
                double h, std::vector<S>& out) const;
  // The block's stage Y = z + hg F(Y), solved in double and carried at S.
  template <class Sys, class S>
  std::vector<S> implicit_stage(const Sys& sys, const std::vector<S>& z,
                                double hg, double t) const;
  template <class Sys>
  std::vector<double> newton(const Sys& sys, const std::vector<double>& z,
                             double hg, double t) const;
  static double stage_time(int i, double time, double h) {
    return time + c[i] * h;
  }
  // The largest magnitude, not a number where any entry is not.
  static double largest(const std::vector<double>& x);

  size_t size = 0;
  std::vector<state_type> k{6}, stiff_k{6};
  std::vector<double> inputs_start_, inputs_end_;
  Step<block_alone<System>> alone_step_;

  static constexpr double AE[6][6] = {
    {0, 0, 0, 0, 0, 0},
    {1.0 / 2.0, 0, 0, 0, 0, 0},
    {13861.0 / 62500.0, 6889.0 / 62500.0, 0, 0, 0, 0},
    {-116923316275.0 / 2393684061468.0, -2731218467317.0 / 15368042101831.0,
     9408046702089.0 / 11113171139209.0, 0, 0, 0},
    {-451086348788.0 / 2902428689909.0, -2682348792572.0 / 7519795681897.0,
     12662868775082.0 / 11960479115383.0, 3355817975965.0 / 11060851509271.0,
     0, 0},
    {647845179188.0 / 3216320057751.0, 73281519250.0 / 8382639484533.0,
     552539513391.0 / 3454668386233.0, 3354512671639.0 / 8306763924573.0,
     4040.0 / 17871.0, 0}};
  static constexpr double AI[6][6] = {
    {0, 0, 0, 0, 0, 0},
    {1.0 / 4.0, 1.0 / 4.0, 0, 0, 0, 0},
    {8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0, 0, 0, 0},
    {5012029.0 / 34652500.0, -654441.0 / 2922500.0, 174375.0 / 388108.0,
     1.0 / 4.0, 0, 0},
    {15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
     730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 1.0 / 4.0, 0},
    {82889.0 / 524892.0, 0, 15625.0 / 83664.0, 69875.0 / 102672.0,
     -2260.0 / 8211.0, 1.0 / 4.0}};
  // Both tableaus share the solution's weights and the embedded ones.
  static constexpr double b[6] = {82889.0 / 524892.0, 0, 15625.0 / 83664.0,
                                  69875.0 / 102672.0, -2260.0 / 8211.0,
                                  1.0 / 4.0};
  static constexpr double d[6] = {
    4586570599.0 / 29645900160.0, 0, 178811875.0 / 945068544.0,
    814220225.0 / 1159782912.0, -3700637.0 / 11593932.0, 61727.0 / 225920.0};
  static constexpr double c[6] = {0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0,
                                  17.0 / 20.0, 1.0};
  // The block alone's tolerance, relative and then absolute.
  static constexpr double block_tol = 1e-9;
  static constexpr double block_tol_abs = 1e-4 * block_tol;
};

template <class System>
void ArkStep<System>::resize(size_t size_) {
  size = size_;
  for (state_type& stage_rate : k) {
    stage_rate.resize(size);
  }
}

template <class System>
void ArkStep<System>::step(System& system, solved_row& solved, double time,
                           double step_size, state_type& y, state_type& yerr,
                           const state_type& dydt_in, state_type& dydt_out) {
  const double h = step_size;
  // First-same-as-last: the first stage is explicit, at the step's start.
  std::copy(dydt_in.begin(), dydt_in.end(), k[0].begin());
  take_step(system, solved, time, h, y, k, stiff_k, y);
  ode::derivs(system, y, dydt_out, time + h, solved.at_state);

  // The pair's embedded estimate; block_error replaces it on the stiff block.
  for (size_t q = 0; q < size; ++q) {
    yerr[q] = h * ((b[0] - d[0]) * k[0][q] + (b[2] - d[2]) * k[2][q] +
                   (b[3] - d[3]) * k[3][q] + (b[4] - d[4]) * k[4][q] +
                   (b[5] - d[5]) * k[5][q]);
  }
}

template <class System>
template <class Sys, class Row, class S>
void ArkStep<System>::take_step(Sys& sys, Row& solved, double time, double h,
                                const std::vector<S>& y0,
                                std::vector<std::vector<S>>& k_,
                                std::vector<std::vector<S>>& stiff,
                                std::vector<S>& y_end) const {
  const auto [first, n] = sys.stiff_block();
  if (first + n > size) {
    util::stop("ArkStep: the stiff block reaches past the state");
  }
  // Each stage's stiff rates, which its implicit terms carry onward.
  for (std::vector<S>& stage_rate : stiff) {
    stage_rate.resize(n);
  }
  std::vector<S> stage(size), z(n);
  std::vector<S> block(y0.begin() + static_cast<std::ptrdiff_t>(first),
                       y0.begin() + static_cast<std::ptrdiff_t>(first + n));
  sys.stiff_rates(stage_time(0, time, h), block, stiff[0]);
  for (int i = 1; i < 6; ++i) {
    const double t_i = stage_time(i, time, h);
    explicit_stage(i, y0, k_, h, stage);
    // The explicit sum carried every rate at the explicit tableau's weights; the
    // block's stiff rates take the implicit tableau's instead.
    for (size_t q = 0; q < n; ++q) {
      z[q] = stage[first + q];
      for (int j = 0; j < i; ++j) {
        z[q] = z[q] + h * (AI[i][j] - AE[i][j]) * stiff[j][q];
      }
    }
    block = implicit_stage(sys, z, h * AI[i][i], t_i);
    std::copy(block.begin(), block.end(),
              stage.begin() + static_cast<std::ptrdiff_t>(first));
    ode::derivs(sys, stage, k_[i], t_i, solved.stages[i - 1]);
    sys.stiff_rates(t_i, block, stiff[i]);
  }
  step_end(y0, k_, h, y_end);
}

template <class System>
template <class S>
void ArkStep<System>::explicit_stage(int i, const std::vector<S>& y0,
                                     const std::vector<std::vector<S>>& k_,
                                     double h, std::vector<S>& out) const {
  if (i == 1) {
    for (size_t q = 0; q < size; ++q) {
      out[q] = y0[q] + AE[1][0] * h * k_[0][q];
    }
    return;
  }
  for (size_t q = 0; q < size; ++q) {
    S combination = AE[i][0] * k_[0][q];
    for (int m = 1; m < i; ++m) {
      combination += AE[i][m] * k_[m][q];
    }
    out[q] = y0[q] + h * combination;
  }
}

template <class System>
template <class S>
void ArkStep<System>::step_end(const std::vector<S>& y0,
                               const std::vector<std::vector<S>>& k_, double h,
                               std::vector<S>& out) const {
  for (size_t q = 0; q < size; ++q) {
    const S combination = b[0] * k_[0][q] + b[2] * k_[2][q] + b[3] * k_[3][q] +
                          b[4] * k_[4][q] + b[5] * k_[5][q];
    out[q] = y0[q] + h * combination;
  }
}

template <class System>
template <class Sys, class S>
std::vector<S> ArkStep<System>::implicit_stage(const Sys& sys,
                                               const std::vector<S>& z,
                                               double hg, double t) const {
  const size_t n = z.size();
  std::vector<double> z_value(n);
  for (size_t q = 0; q < n; ++q) {
    z_value[q] = util::to_passive(z[q]);
  }
  const std::vector<double> y_star = newton(sys, z_value, hg, t);
  if constexpr (std::is_same_v<S, double>) {
    return y_star;
  } else {
    // The residual G(Y) = Y - z - hg F(Y), whose Jacobian is I - hg J.
    std::vector<double> dGdY(n * n);
    sys.stiff_jacobian(t, y_star, dGdY);
    for (size_t r = 0; r < n; ++r) {
      for (size_t col = 0; col < n; ++col) {
        dGdY[r * n + col] = (r == col ? 1.0 : 0.0) - hg * dGdY[r * n + col];
      }
    }
    return implicit_values<S>(
        y_star, std::move(dGdY),
        [&](const std::vector<S>& Y) -> std::vector<S> {
          std::vector<S> F(n), G(n);
          sys.stiff_rates(t, Y, F);
          for (size_t q = 0; q < n; ++q) {
            G[q] = Y[q] - z[q] - hg * F[q];
          }
          return G;
        });
  }
}

// Newton from z, each step halved until the residual's largest component falls; a
// stage it cannot reach raises DomainError, which rejects the step.
template <class System>
template <class Sys>
std::vector<double> ArkStep<System>::newton(const Sys& sys,
                                            const std::vector<double>& z,
                                            double hg, double t) const {
  const size_t n = z.size();
  std::vector<double> y = z, f(n), g(n), m(n * n), dy(n), rhs(n), y_next(n),
                      g_next(n);
  std::vector<size_t> pivots;
  sys.stiff_rates(t, y, f);
  for (size_t q = 0; q < n; ++q) {
    g[q] = -hg * f[q];
  }
  for (int iteration = 0; iteration < 30; ++iteration) {
    sys.stiff_jacobian(t, y, m);
    for (size_t r = 0; r < n; ++r) {
      for (size_t col = 0; col < n; ++col) {
        m[r * n + col] = (r == col ? 1.0 : 0.0) - hg * m[r * n + col];
      }
    }
    linalg::lu_decompose(m, n, pivots);
    for (size_t q = 0; q < n; ++q) {
      rhs[q] = -g[q];
    }
    linalg::lu_solve(m, n, pivots, rhs, dy);
    if (largest(dy) < 1e-12) {
      for (size_t q = 0; q < n; ++q) {
        y[q] = y[q] + dy[q];
      }
      return y;
    }
    double a = 1.0;
    while (true) {
      for (size_t q = 0; q < n; ++q) {
        y_next[q] = y[q] + a * dy[q];
      }
      sys.stiff_rates(t, y_next, f);
      for (size_t q = 0; q < n; ++q) {
        g_next[q] = y_next[q] - z[q] - hg * f[q];
      }
      if (largest(g_next) < largest(g)) {
        break;
      }
      a = a / 2.0;
      if (a < 0x1p-10) {
        break;
      }
    }
    if (a < 0x1p-10) {
      break;
    }
    std::swap(y, y_next);
    std::swap(g, g_next);
  }
  util::stop_domain("ArkStep: Newton did not reach the stiff block's stage at "
                    "t = " + util::format_double(t));
}

template <class System>
double ArkStep<System>::largest(const std::vector<double>& x) {
  double ret = 0.0;
  for (const double v : x) {
    const double a = std::abs(v);
    if (std::isnan(a)) {
      return a;
    }
    if (a > ret) {
      ret = a;
    }
  }
  return ret;
}

template <class System>
void ArkStep<System>::start_inputs(const System& system) {
  system.stiff_inputs(inputs_start_);
}

template <class System>
void ArkStep<System>::block_error(const System& system, double time,
                                  double step_size, const state_type& y_start,
                                  const state_type& y_end, state_type& yerr) {
  const auto [first, n] = system.stiff_block();
  if (n == 0) {
    return;
  }
  system.stiff_inputs(inputs_end_);
  util::check_length(inputs_start_.size(), n);
  util::check_length(inputs_end_.size(), n);
  block_alone<System> alone{&system, time, step_size, &inputs_start_,
                            &inputs_end_, std::vector<double>(n),
                            std::vector<double>(n), std::vector<double>(n)};
  alone_step_.resize(n);
  typename Step<block_alone<System>>::solved_row row{};

  std::vector<double> y(n), rates(n), y_next(n), error(n), rates_next(n);
  for (size_t q = 0; q < n; ++q) {
    y[q] = util::to_passive(y_start[first + q]);
  }
  const double t_end = time + step_size;
  double s = time, h = step_size;
  ode::derivs(alone, y, rates, s);
  while (s < t_end) {
    const bool last = s + h >= t_end;
    const double h_try = last ? t_end - s : h;
    y_next = y;
    alone_step_.step(alone, row, s, h_try, y_next, error, rates, rates_next);
    double r = 0.0;
    bool finite = true;
    for (size_t q = 0; q < n; ++q) {
      const double v = std::abs(error[q]) /
                       (block_tol * std::abs(y_next[q]) + block_tol_abs);
      if (!std::isfinite(v)) {
        finite = false;
      } else if (v > r) {
        r = v;
      }
    }
    if (!finite) {
      r = 1e10;
    }
    if (r > 1.1 && h_try > 1e-12) {
      h = h_try * std::max(0.2, 0.9 / std::pow(r, 1.0 / 5.0));
      continue;
    }
    s = last ? t_end : s + h_try;
    std::swap(y, y_next);
    std::swap(rates, rates_next);
    if (!last) {
      h = h_try * std::min(5.0, std::max(0.2, 0.9 / std::pow(std::max(r, 1e-300),
                                                               1.0 / 5.0)));
    }
  }
  for (size_t q = 0; q < n; ++q) {
    yerr[first + q] = y_end[first + q] - y[q];
  }
}

template <class System>
void ArkStep<System>::step_adjoint(active_system<System>& active,
                                   const solved_values& first, double first_time,
                                   const solved_row& solved, double time,
                                   double step_size, const state_type& y,
                                   const adjoint_rows& lambda_out,
                                   adjoint_rows& lambda_in,
                                   adjoint_rows& parameter_adjoint) {
  using scalar = active_scalar<double>;
  const double h = step_size;
  if (lambda_out.empty()) {
    util::stop("step_adjoint: needs at least one seed");
  }
  util::check_length(y.size(), size);

  auto whole_step = [&](auto& sys,
                        typename std::vector<scalar>::const_iterator x,
                        std::vector<scalar>& y_end) -> void {
    const std::vector<scalar> y0(x, x + static_cast<std::ptrdiff_t>(size));
    std::vector<std::vector<scalar>> rate(6, std::vector<scalar>(size)),
        stiff(6);
    // k1 repeats the evaluation the row below recorded at this state.
    ode::derivs(sys, y0, rate[0], first_time, first);
    take_step(sys, solved, time, h, y0, rate, stiff, y_end);
    recorded_rates += 6;
  };

  ode::state_and_parameter_adjoints(active, y, lambda_out, whole_step, lambda_in,
                                    parameter_adjoint);
}

}
}

#endif
