// -*-c++-*-
#ifndef ODELIA_ODE_STEP_MRI_IMPL_HPP_
#define ODELIA_ODE_STEP_MRI_IMPL_HPP_

// Out-of-line definition of MriStep::step. Split from ode_step_mri.hpp so it can
// be included after SolverInternal is defined: the macro step sub-cycles the fast
// block through SolverInternal (via mri_macro_step in mri.hpp), which therefore
// must already exist. Included at the bottom of ode_solver_internal.hpp.

#include <odelia/ode_step_mri.hpp>
#include <odelia/mri.hpp>

namespace odelia {
namespace ode {

// The body shared by the MRI steppers: split [slow | fast], run one macro step
// with the given coupling + inner, copy back, report no macro-level error (the
// macro grid is the fixed forcing-kink grid), and sync rates/state for history.
template <class System, class Subcycle>
void mri_step_run(System& system, const MRICoupling& coupling,
                  const OdeControl& inner_control, double time, double step_size,
                  std::vector<typename System::value_type>& y,
                  std::vector<typename System::value_type>& yerr,
                  std::vector<typename System::value_type>& dydt_out,
                  const Subcycle& subcycle) {
  using value_type = typename System::value_type;
  const size_t ns = system.slow_size();
  std::vector<value_type> x(y.begin(), y.begin() + ns), u(y.begin() + ns, y.end());
  MRISchedule sched;
  mri_macro_step(system, coupling, inner_control, time, step_size,
                 x, u, sched, /*replay=*/false, subcycle);
  std::copy(x.begin(), x.end(), y.begin());
  std::copy(u.begin(), u.end(), y.begin() + ns);
  std::fill(yerr.begin(), yerr.end(), value_type(0.0));
  ode::derivs(system, y, dydt_out, time + step_size);
}

template <class System>
void MriStep<System>::step(System& system, double time, double step_size,
                           state_type& y, state_type& yerr,
                           const state_type& /*dydt_in*/, state_type& dydt_out) {
  if constexpr (supported) {
    MRICoupling coupling = mri_erk33a();
    // Select the fast-block inner: the exact-flow split (Lever 1) when the
    // System opts in at runtime, else the adaptive black-box inner. Both are
    // instantiated only for a System that provides mri_split() (which implies
    // the analytic_flow / residual_rhs hooks the split needs).
    if constexpr (mri_wants_split<System>::value) {
      if (system.mri_split())
        mri_step_run(system, coupling, inner_control, time, step_size, y, yerr,
                     dydt_out, SplitSubcycle{});
      else
        mri_step_run(system, coupling, inner_control, time, step_size, y, yerr,
                     dydt_out, AdaptiveSubcycle{});
    } else {
      mri_step_run(system, coupling, inner_control, time, step_size, y, yerr,
                   dydt_out, AdaptiveSubcycle{});
    }
  }
}

template <class System>
void MriUptakeStep<System>::step(System& system, double time, double step_size,
                                 state_type& y, state_type& yerr,
                                 const state_type& /*dydt_in*/,
                                 state_type& dydt_out) {
  if constexpr (supported) {
    // Freeze cohorts over the leg and sub-cycle the soil against the affine-
    // refreshed uptake with a trust monitor. The slow (cohort) block is advanced
    // by a 3rd-order MRI-GARK coupling: under time-varying forcing a 1st-order
    // (forward-Euler) slow advance leaves an O(H) offspring bias (~12% at a weekly
    // leg under seasonal rainfall); kutta3 cuts that to <0.1% at the same leg size
    // while keeping the cohort-solve reduction (Slice 4). Production monitor
    // (oracle=false); tol / nmicro from the System.
    MRICoupling coupling = mri_kutta3();
    UptakeSubcycle sub{system.mri_uptake_tol(), system.mri_uptake_nmicro(),
                       /*oracle=*/false};
    mri_step_run(system, coupling, inner_control, time, step_size, y, yerr,
                 dydt_out, sub);
  }
}

}
}

#endif
