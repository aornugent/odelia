// -*-c++-*-
#ifndef ODELIA_ODE_STEP_HPP_
#define ODELIA_ODE_STEP_HPP_

#include <vector>
#include <cstddef>
#include <XAD/XAD.hpp>
#include <odelia/adjoint.hpp>
#include <odelia/ode_interface.hpp>
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
  // way back. The active System the batched form is handed is this call's, where a sweep
  // hands the same one to every step.
  void step_adjoint(System& system,
                    double time, double step_size,
                    const state_type &y,
                    const state_type &lambda_out,
                    state_type &lambda_in,
                    std::vector<double>& parameter_adjoint);

  // The step transposed for several seeds at once. The six rate evaluations that
  // rebuild the stage run ONCE, and each stage's transpose is recorded once and
  // swept per seed.
  template <class ActiveSystem>
  void step_adjoint_batched(System& system,
                            double time, double step_size,
                            const state_type &y,
                            const std::vector<state_type> &lambda_out,
                            std::vector<state_type> &lambda_in,
                            std::vector<std::vector<double>>& parameter_adjoint,
                            ActiveSystem& active_system);

  // How many stage transposes have been swept, seeds counted separately. A row
  // that acts once per stage is multiplied by this, and a row correct per
  // evaluation and wrong in its multiplier is a different failure from a wrong
  // row, which neither a tangent nor a sweep can see.
  std::size_t stage_sweeps = 0;

  void derivs(System& system, const state_type& y, state_type& dydt, double t, int index) {
    return ode::derivs(system, y, dydt, t, index);
  }

  // These are defined in rkck_type
  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // Y_i for stage i, into ytmp: y at stage 0, and y plus the tableau's
  // combination of the earlier stage rates above that. step()'s own arithmetic,
  // term for term, so a rebuilt stage state is bit-identical to the one step()
  // stepped through.
  void stage_state(int i, const state_type& y, double h);
  double stage_time(int i, double time, double h) const;
  const double* stage_row(int i) const;

  // One reverse traversal of the stages, shared by every System: descending, so
  // a stage's rate adjoint is complete before it is used, since stage i reads
  // only the rates of stages before it. The stage state is rebuilt ONCE however
  // many seeds ride along, and `sweep_stage` is handed all of them together so
  // the stage's transpose is recorded once and swept per seed.
  template <class SweepStage>
  void sweep_stages(double time, double h, const state_type& y,
                    std::vector<std::vector<state_type>>& lambda_k,
                    std::vector<state_type>& lambda_in,
                    SweepStage sweep_stage);

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
void Step<System>::stage_state(int i, const state_type& y, double h) {
  if (i == 0) {
    std::copy(y.begin(), y.end(), ytmp.begin());
    return;
  }
  // Stage 1 keeps step()'s grouping of the single term: b21 * h * k1 and
  // h * (b21 * k1) round differently.
  if (i == 1) {
    for (size_t q = 0; q < size; ++q) {
      ytmp[q] = y[q] + b21 * h * k1[q];
    }
    return;
  }
  const state_type* const k[] = {&k1, &k2, &k3, &k4, &k5, &k6};
  const double* const b = stage_row(i);
  for (size_t q = 0; q < size; ++q) {
    // Summed in ascending stage, then one h, as step() sums it. Cash-Karp's
    // rows are dense, so this is a sum over every earlier stage rather than a
    // term for the immediate predecessor.
    value_type combination = b[0] * k1[q];
    for (int m = 1; m < i; ++m) {
      combination += b[m] * (*k[m])[q];
    }
    ytmp[q] = y[q] + h * combination;
  }
}

template <class System>
template <class SweepStage>
void Step<System>::sweep_stages(double time, double h, const state_type& y,
                                std::vector<std::vector<state_type>>& lambda_k,
                                std::vector<state_type>& lambda_in,
                                SweepStage sweep_stage) {
  const std::size_t n_seed = lambda_in.size();
  std::vector<state_type> lambda_stage(n_seed, state_type(size));
  std::vector<state_type> lambda_rate(n_seed, state_type(size));
  for (int i = 5; i >= 0; --i) {
    stage_state(i, y, h);
    for (std::size_t m = 0; m < n_seed; ++m) {
      lambda_rate[m] = lambda_k[m][i];
    }
    sweep_stage(i, stage_time(i, time, h), ytmp, lambda_rate, lambda_stage);
    for (std::size_t m = 0; m < n_seed; ++m) {
      // The stage state is y plus the earlier rates, so its adjoint splits over
      // both.
      for (size_t q = 0; q < size; ++q) {
        lambda_in[m][q] += lambda_stage[m][q];
      }
      if (i == 0) {
        continue;
      }
      const double* const b = stage_row(i);
      for (int l = 0; l < i; ++l) {
        for (size_t q = 0; q < size; ++q) {
          lambda_k[m][l][q] += h * b[l] * lambda_stage[m][q];
        }
      }
    }
  }
}

// lambda_in[m] = (d y_end / d y)^T lambda_out[m] for the one step step() takes from
// y. The step's own stage rates go back into k1..k6, since the stage states are y
// plus the RKCK combination of those rates and the caller keeps only y. The rebuild
// runs forward, because Y_i needs k_1 ... k_{i-1}; the sweep runs backward, because
// a stage's rate adjoint is complete only once the stages above it are swept. Every
// seed rides the same trajectory, so both run ONCE however many are carried.
//
// Each stage is recorded once on the tape this stepper keeps and swept once per
// seed -- the state and the active System's parameters in one recording, so a stage the
// parameters reach carries their rows too. The recording is derivs(), which is
// what the forward pass calls, so no System writes a transpose of its own.
template <class System>
template <class ActiveSystem>
void Step<System>::step_adjoint_batched(System& system,
                                        double time, double step_size,
                                        const state_type &y,
                                        const std::vector<state_type> &lambda_out,
                                        std::vector<state_type> &lambda_in,
                                        std::vector<std::vector<double>>& parameter_adjoint,
                                        ActiveSystem& active_system) {
  using active_type = active_scalar<double>;
  static_assert(Rebindable<System, active_type>,
                "step_adjoint_batched needs the System's rebind_from() hook to "
                "lift it to the adjoint scalar");
  const double h = step_size;
  const std::size_t n_seed = lambda_out.size();
  if (n_seed == 0) {
    util::stop("step_adjoint_batched: needs at least one seed");
  }

  // Six rate evaluations, k1 among them: first-same-as-last carries k1 into
  // step() as the previous step's dydt_out, which a reverse traversal has not
  // rebuilt, so it is re-derived here at this step's own start state.
  state_type* const k[] = {&k1, &k2, &k3, &k4, &k5, &k6};
  for (int i = 0; i < 6; ++i) {
    stage_state(i, y, h);
    if (i == 0) {
      ode::derivs(system, ytmp, *k[i], time);
    } else {
      ode::derivs(system, ytmp, *k[i], stage_time(i, time, h), i - 1);
    }
  }

  // The direct term of y_end = y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6), then the
  // rate adjoints it seeds. k2 and k5 enter y_end only through later stages.
  lambda_in.assign(n_seed, state_type(size));
  std::vector<std::vector<state_type>> lambda_k(
    n_seed, std::vector<state_type>(6, state_type(size, value_type(0.0))));
  for (std::size_t m = 0; m < n_seed; ++m) {
    lambda_in[m].assign(lambda_out[m].begin(), lambda_out[m].end());
    for (size_t q = 0; q < size; ++q) {
      lambda_k[m][0][q] = h * c1 * lambda_out[m][q];
      lambda_k[m][2][q] = h * c3 * lambda_out[m][q];
      lambda_k[m][3][q] = h * c4 * lambda_out[m][q];
      lambda_k[m][5][q] = h * c6 * lambda_out[m][q];
    }
  }

  sweep_stages(time, h, y, lambda_k, lambda_in,
               [&](int i, double stage_t, const state_type& stage,
                   const std::vector<state_type>& lambda_rate,
                   std::vector<state_type>& lambda_stage) -> void {
    ode::rates_adjoint(adjoint_tape.get(), system, active_system, stage, stage_t, i - 1,
                       lambda_rate, lambda_stage, parameter_adjoint);
    stage_sweeps += n_seed;
  });

  // Rebuilding the stages walked the System through their states and left it at
  // the last of them, so put it back where the step started. A caller sweeping
  // steps backwards reads the System between calls.
  ode::internal::set_ode_state(system, y, time);
}

template <class System>
void Step<System>::step_adjoint(System& system,
                                double time, double step_size,
                                const state_type &y,
                                const state_type &lambda_out,
                                state_type &lambda_in,
                                std::vector<double>& parameter_adjoint) {
  using active_type = active_scalar<double>;
  static_assert(Rebindable<System, active_type>,
                "step_adjoint needs the System's rebind_from() hook to lift it to "
                "the adjoint scalar");
  auto active_system = system.template rebind_from<active_type>();
  // An empty accumulator starts from zero rather than meaning the rows are not
  // wanted: they come back either way, and a caller that ignores them has ignored
  // something it was handed.
  if (parameter_adjoint.empty()) {
    parameter_adjoint.assign(active_system.ad_parameters().size(), 0.0);
  }
  std::vector<state_type> seed(1, lambda_out), swept;
  std::vector<std::vector<double>> rows(1, std::move(parameter_adjoint));
  step_adjoint_batched(system, time, step_size, y, seed, swept, rows, active_system);
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
