// -*-c++-*-
#ifndef ODELIA_ODE_STEP_HPP_
#define ODELIA_ODE_STEP_HPP_

#include <vector>
#include <cstddef>
#include <XAD/XAD.hpp>
#include <odelia/adjoint.hpp>
#include <odelia/ode_interface.hpp>

namespace odelia {
namespace ode {

template <class System>
class Step {
public:
  // Extract scalar type from System using traits
  using value_type = typename System::value_type;
  using state_type = std::vector<value_type>;
  
  void resize(size_t size_);
  size_t order() const;
  void step(System& system,
            double time, double step_size,
	    state_type &y,
	    state_type &yerr,
	    const state_type &dydt_in,
	    state_type &dydt_out);

  // One seed of the step below, packed into a batch of one and unpacked on the
  // way back.
  void step_adjoint(System& system,
                    double time, double step_size,
                    const state_type &y,
                    const state_type &lambda_out,
                    state_type &lambda_in,
                    std::vector<double>& parameter_adjoint);

  // The step transposed for several seeds at once: one recording of the whole
  // step, swept once per seed.
  void step_adjoint_batched(System& system,
                            double time, double step_size,
                            const state_type &y,
                            const std::vector<state_type> &lambda_out,
                            std::vector<state_type> &lambda_in,
                            std::vector<std::vector<double>>& parameter_adjoint);

  // Rate evaluations the sweeps since the last clear have recorded: six a step,
  // whatever the seed count, because the step is recorded once and swept per
  // seed. A term entering once a step where it belongs once a stage divides this
  // by six, and no gradient check can see that, because a tangent and a sweep
  // apply the same multiplier.
  std::size_t recorded_rates = 0;

  void derivs(System& system, const state_type& y, state_type& dydt, double t, int index) {
    return ode::derivs(system, y, dydt, t, index);
  }

  // These are defined in rkck_type
  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // Y_i for stage i, into `out`: y at stage 0, and y plus the tableau's
  // combination of the earlier stage rates above that. step()'s own arithmetic,
  // term for term, at whatever scalar the caller holds its rates in.
  template <class S>
  void stage_state(int i, const std::vector<S>& y,
                   const std::vector<std::vector<S>>& k, double h,
                   std::vector<S>& out) const;
  double stage_time(int i, double time, double h) const;
  const double* stage_row(int i) const;

  // Intermediate storage, representing state (was GSL rkck_state_t)
  size_t size;
  state_type k1, k2, k3, k4, k5, k6, ytmp;

  // The tape the stage transposes record on, one for every step a sweep walks.
  scratch_tape adjoint_tape;

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
};

template <class System>
void Step<System>::resize(size_t size_) {
  size = size_;
  k1.resize(size);
  k2.resize(size);
  k3.resize(size);
  k4.resize(size);
  k5.resize(size);
  k6.resize(size);
  ytmp.resize(size);
}

template <class System>
size_t Step<System>::order() const {
  // In GSL, comment says "FIXME: should this be 4?"
  return 5;
}

// Record the per-RK-stage field value on a System that keeps one; a no-op otherwise.
template <typename System>
void record_stage(System& system, int rk_step) {
  if constexpr (ReplaysField<System>) {
    system.record_stage(rk_step);
  }
}


// Think carefully about ownership of data, draw a diagram, and go
// from there.
template <class System>
void Step<System>::step(System& system,
                        double time, double step_size,
                        state_type &y,
                        state_type &yerr,
                        const state_type &dydt_in,
                        state_type &dydt_out) {
  const double h = step_size; // Historical reasons.

  // k1 step:
  std::copy(dydt_in.begin(), dydt_in.end(), k1.begin());
  for (size_t i = 0; i < size; ++i) {
    ytmp[i] = y[i] + b21 * h * k1[i];
  }

  // k2 step:
  derivs(system, ytmp, k2, time + ah[0] * h, 0);
  record_stage(system, 0);
  
  for (size_t i = 0; i < size; ++i) {
    ytmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);
  }

  // k3 step:
  derivs(system, ytmp, k3, time + ah[1] * h, 1);
  record_stage(system, 1);

  for (size_t i = 0; i < size; ++i) {
    ytmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);
  }

  // k4 step:
  derivs(system, ytmp, k4, time + ah[2] * h, 2);
  record_stage(system, 2);

  for (size_t i = 0; i < size; ++i) {
    ytmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] +
			  b5[3] * k4[i]);
  }

  // k5 step
  derivs(system, ytmp, k5, time + ah[3] * h, 3);
  record_stage(system, 3);

  for (size_t i = 0; i < size; ++i) {
    ytmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] +
			  b6[3] * k4[i] + b6[4] * k5[i]);
  }

  // k6 step and final sum
  derivs(system, ytmp, k6, time + ah[4] * h, 4);
  record_stage(system, 4);

  for (size_t i = 0; i < size; ++i) {
    // GSL does this in two steps, but not sure why.
    const value_type d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c6 * k6[i];
    y[i] += h * d_i;
  }

  // Evaluate dydt_out.
  derivs(system, y, dydt_out, time + h, 5);
  record_stage(system, 5);

  // Difference between 4th and 5th order, for error calculations
  for (size_t i = 0; i < size; ++i) {
    yerr[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] +
		   ec[5] * k5[i] + ec[6] * k6[i]);
  }
}

// The tableau row stage i's state combines the earlier stage rates with.
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
  // Stage 1 keeps step()'s grouping of the single term: b21 * h * k1 and
  // h * (b21 * k1) round differently.
  if (i == 1) {
    for (size_t q = 0; q < size; ++q) {
      out[q] = y[q] + b21 * h * k[0][q];
    }
    return;
  }
  const double* const b = stage_row(i);
  for (size_t q = 0; q < size; ++q) {
    // Summed in ascending stage, then one h, as step() sums it. Cash-Karp's
    // rows are dense, so this is a sum over every earlier stage rather than a
    // term for the immediate predecessor.
    S combination = b[0] * k[0][q];
    for (int m = 1; m < i; ++m) {
      combination += b[m] * k[m][q];
    }
    out[q] = y[q] + h * combination;
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
void Step<System>::step_adjoint_batched(System& system,
                                        double time, double step_size,
                                        const state_type &y,
                                        const std::vector<state_type> &lambda_out,
                                        std::vector<state_type> &lambda_in,
                                        std::vector<std::vector<double>>& parameter_adjoint) {
  using scalar = active_scalar<double>;
  const double h = step_size;
  if (lambda_out.empty()) {
    util::stop("step_adjoint_batched: needs at least one seed");
  }

  auto whole_step = [&](auto& active_system,
                        typename std::vector<scalar>::const_iterator x,
                        std::vector<scalar>& y_end) -> void {
    const std::vector<scalar> y0(x, x + static_cast<std::ptrdiff_t>(size));
    std::vector<std::vector<scalar>> k(6, std::vector<scalar>(size));
    std::vector<scalar> stage(size);
    // k1 is re-derived at this step's own start state: first-same-as-last hands
    // step() the previous step's dydt_out, which a reverse traversal has not
    // rebuilt.
    ode::derivs(active_system, y0, k[0], time);
    for (int i = 1; i < 6; ++i) {
      stage_state(i, y0, k, h, stage);
      ode::derivs(active_system, stage, k[i], stage_time(i, time, h), i - 1);
    }
    // y_end = y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6). k2 and k5 reach it only
    // through the later stages.
    for (size_t q = 0; q < size; ++q) {
      const scalar d_q = c1 * k[0][q] + c3 * k[2][q] + c4 * k[3][q] + c6 * k[5][q];
      y_end[q] = y0[q] + h * d_q;
    }
  };

  ode::state_and_parameter_adjoints(adjoint_tape.get(), system, y, lambda_out,
                                    whole_step, lambda_in, parameter_adjoint);
  recorded_rates += 6;

  // The stages are the recording's, so the double System was not walked through
  // them; it is put where the step started for the caller that reads it between
  // steps.
  ode::internal::set_ode_state(system, y, time);
}

template <class System>
void Step<System>::step_adjoint(System& system,
                                double time, double step_size,
                                const state_type &y,
                                const state_type &lambda_out,
                                state_type &lambda_in,
                                std::vector<double>& parameter_adjoint) {
  // An empty accumulator starts from zero rather than meaning the rows are not
  // wanted: they come back either way, and a caller that ignores them has ignored
  // something it was handed.
  if (parameter_adjoint.empty()) {
    parameter_adjoint.assign(system.ad_parameters().size(), 0.0);
  }
  std::vector<state_type> seed(1, lambda_out), swept;
  std::vector<std::vector<double>> rows(1, std::move(parameter_adjoint));
  step_adjoint_batched(system, time, step_size, y, seed, swept, rows);
  parameter_adjoint = std::move(rows[0]);
  lambda_in = std::move(swept[0]);
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

}
}

#endif
