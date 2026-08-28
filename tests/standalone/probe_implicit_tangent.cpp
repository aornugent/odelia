// PROBE: implicit_value at the four scalars a caller can ask for, against an
// answer that is known rather than differenced. F(y; p) = y^3 - p places
// y* = p^(1/3), so dy*/dp = 1/(3 y*^2) and d2y*/dp2 = -(2/9) p^(-5/3), exactly.
//
// The FOURTH case is the one that matters and was missing: the nested
// forward-over-reverse scalar, which is what a caller asking for a curvature uses
// (plant's Leaf::collar_condition). implicit_value carries a second correction
// specifically so that scalar gets the true curvature rather than the linearised
// theorem's -- gated on `SecondOrder<S>` -- and that correction was asserted in a
// comment and measured nowhere. A wrong curvature here is a wrong SIGN downstream:
// TF24 refuses an operating point whose profit curvature is not negative.
//
//   make -C tests/standalone probe_implicit_tangent

#include <odelia/implicit_node.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/tangent.hpp>

#include <cmath>
#include <cstdio>

using odelia::ode::active_scalar;
using odelia::ode::adjoint_tape;
using odelia::ode::tangent_scalar;
using odelia::ode::seed_direction;
using odelia::ode::derivative_along;
using odelia::ode::directional_adjoint;
using odelia::ode::directional_adjoint_scalar;
using odelia::ode::directional_adjoint_tape;
using odelia::ode::seed_inner_direction;

namespace {

// The residual, and the slope it knows: d/dy (y^3 - p) = 3 y^2.
template <class S>
S cube_root_at(double y_star, const S& p) {
  return odelia::implicit_value<S>(
      y_star, 3.0 * y_star * y_star,
      [&](const S& y) -> S { return y * y * y - p; });
}

int failures = 0;

void check(const char* what, double got, double want, double tol) {
  const double err = std::abs(got - want) / (std::abs(want) + 1e-300);
  const bool ok = err <= tol;
  if (!ok) ++failures;
  std::printf("  %-28s got %.17g  want %.17g  rel %.3e  %s\n", what, got, want,
              err, ok ? "ok" : "FAIL");
}

}  // namespace

int main() {
  const double p0 = 8.0;
  const double y0 = std::cbrt(p0);
  const double want = 1.0 / (3.0 * y0 * y0);

  std::printf("implicit_value: one residual, its own slope supplied\n\n");
  check("double: value", cube_root_at<double>(y0, p0), y0, 0.0);

  {
    using S = tangent_scalar<double>;
    S p = p0;
    seed_direction(p, 1.0);
    const S y = cube_root_at<S>(y0, p);
    check("forward: value", odelia::util::to_passive(y), y0, 0.0);
    check("forward: dy/dp", derivative_along(y), want, 1e-15);
  }

  {
    using S = active_scalar<double>;
    adjoint_tape<double> tape;  // constructing one activates it
    S p = p0;
    tape.registerInput(p);
    tape.newRecording();
    S y = cube_root_at<S>(y0, p);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();
    check("reverse: value", odelia::util::to_passive(y), y0, 0.0);
    check("reverse: dy/dp", derivative(p), want, 1e-15);
  }

  // The curvature, on the scalar a curvature is actually asked for.
  {
    using S = directional_adjoint_scalar<double>;
    std::printf("\n  SecondOrder<directional_adjoint_scalar<double>> = %s\n",
                odelia::ode::SecondOrder<S> ? "true (correction applied)"
                                            : "FALSE (correction SKIPPED)");
    const double want2 = -(2.0 / 9.0) * std::pow(p0, -5.0 / 3.0);

    directional_adjoint_tape<double> tape;
    tape.clearAll();
    S p = p0;
    tape.registerInput(p);
    tape.newRecording();
    seed_inner_direction(p, 1.0);
    S y = cube_root_at<S>(y0, p);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();
    check("nested: value", odelia::util::to_passive(y), y0, 0.0);
    check("nested: d2y/dp2", directional_adjoint(p), want2, 1e-10);

    // And what the LINEARISED theorem would have given, for contrast: with the
    // denominator held constant the correction is -F_pp/F_y, and F_pp = 0 here,
    // so a first-order-only record reports a curvature of exactly zero. That is
    // the failure mode this case exists to catch -- not a wrong magnitude but a
    // curvature that has lost its sign information entirely.
    std::printf("  %-28s %.17g   (a linearised record reports 0)\n",
                "for contrast, true d2y/dp2", want2);
  }

  std::printf("\n  %s\n", failures == 0 ? "all ok" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
