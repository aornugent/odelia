// -*-c++-*-
#ifndef ODELIA_ODE_STEP_HPP_
#define ODELIA_ODE_STEP_HPP_

#include <vector>
#include <cstddef>
#include <XAD/XAD.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_jacobian.hpp>

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

  void step_adjoint(System& system,
                    double time, double step_size,
                    const state_type &y,
                    const state_type &lambda_out,
                    state_type &lambda_in);

  void derivs(System& system, const state_type& y, state_type& dydt, double t, int index) {
    return ode::derivs(system, y, dydt, t, index);
  }

  // These are defined in rkck_type
  static const bool can_use_dydt_in = true;
  static const bool first_same_as_last = true;

private:
  // Intermediate storage, representing state (was GSL rkck_state_t)
  size_t size;
  state_type k1, k2, k3, k4, k5, k6, ytmp;

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

// Record the per-RK-stage field values on a Replayable System; a no-op otherwise.
template <typename System>
void record_stage(System& system, int rk_step) {
  if constexpr (Replayable<System>) {
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

// lambda_in = (d y_end / d y)^T lambda_out for the one step step() takes from y. The
// step's own stage rates go back into k1..k6, since the stage states are y plus the
// RKCK combination of those rates and the caller keeps only y.
//
// Each stage contributes J_f(stage state)^T times that stage's rate adjoint. Those
// vector-Jacobian products are recorded on a tape of this function's own, on the System
// lifted to the adjoint scalar, so the caller's reverse pass stays in value_type.
template <class System>
void Step<System>::step_adjoint(System& system,
                                double time, double step_size,
                                const state_type &y,
                                const state_type &lambda_out,
                                state_type &lambda_in) {
  static_assert(has_rebind_from<System>::value,
                "step_adjoint needs the System's rebind_from() hook to lift it to the "
                "adjoint scalar");
  using ad = xad::adj<value_type>;
  using active_type = typename ad::active_type;

  const double h = step_size;

  // Retake the step to refill k1..k6; step() overwrites the state it is given, so the
  // start state is read from y throughout below.
  state_type dydt_in(size), dydt_out(size), yerr(size), y_end(y);
  ode::derivs(system, y, dydt_in, time);
  step(system, time, h, y_end, yerr, dydt_in, dydt_out);

  const state_type* const k[] = {&k1, &k2, &k3, &k4, &k5, &k6};
  const double* const b[] = {&b21, b3, b4, b5, b6};  // stage j reads row b[j - 2]

  // The direct term of y_end = y + h * (c1 k1 + c3 k3 + c4 k4 + c6 k6), then the rate
  // adjoints it seeds. k2 and k5 enter y_end only through later stages.
  lambda_in.assign(lambda_out.begin(), lambda_out.end());
  std::vector<state_type> lambda_k(6, state_type(size, value_type(0.0)));
  for (size_t i = 0; i < size; ++i) {
    lambda_k[0][i] = h * c1 * lambda_out[i];
    lambda_k[2][i] = h * c3 * lambda_out[i];
    lambda_k[3][i] = h * c4 * lambda_out[i];
    lambda_k[5][i] = h * c6 * lambda_out[i];
  }

  auto twin = system.template rebind_from<active_type>();
  typename ad::tape_type tape;  // Activates here; deactivates when it goes out of scope.
  state_type lambda_stage(size);

  // Descending, so a stage's rate adjoint is complete before it is swept: stage j reads
  // only the rates of stages before it.
  for (int j = 6; j >= 1; --j) {
    // Fresh values each stage: an input carrying a slot from the previous recording
    // registers as a variable with no dependencies, and its adjoint sweeps to zero.
    std::vector<active_type> stage(size), rates(size);
    for (size_t i = 0; i < size; ++i) {
      // step()'s own arithmetic, so the stage state is bit-identical to the one it
      // stepped through: the rate combination summed in ascending m, then one h.
      value_type s = y[i];
      if (j == 2) {
        s = y[i] + b21 * h * k1[i];
      } else if (j > 2) {
        value_type combination = b[j - 2][0] * k1[i];
        for (int m = 2; m < j; ++m) {
          combination += b[j - 2][m - 1] * (*k[m - 1])[i];
        }
        s = y[i] + h * combination;
      }
      stage[i] = active_type(s);
    }
    tape.registerInputs(stage);
    tape.newRecording();

    if (j == 1) {
      ode::derivs(twin, stage, rates, time);
    } else {
      ode::derivs(twin, stage, rates, time + ah[j - 2] * h, j - 2);
    }
    tape.registerOutputs(rates);
    for (size_t i = 0; i < size; ++i) {
      xad::derivative(rates[i]) = lambda_k[j - 1][i];
    }
    tape.computeAdjoints();
    for (size_t i = 0; i < size; ++i) {
      lambda_stage[i] = xad::derivative(stage[i]);
    }

    // The stage state is y plus the earlier rates, so its adjoint splits over both.
    for (size_t i = 0; i < size; ++i) {
      lambda_in[i] += lambda_stage[i];
      for (int m = 1; m < j; ++m) {
        lambda_k[m - 1][i] += h * b[j - 2][m - 1] * lambda_stage[i];
      }
    }
  }

  // Retaking the step walked the System through the stage states and left it at the
  // last of them, so put it back where the step started. A caller sweeping steps
  // backwards reads the System between calls.
  ode::internal::set_ode_state(system, y, time);
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
