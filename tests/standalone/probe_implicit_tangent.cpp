// PROBE: implicit_value at the three scalars a caller can ask for, against an
// answer that is known rather than differenced. F(y; p) = y^3 - p places
// y* = p^(1/3), so dy*/dp = 1/(3 y*^2) exactly.
//
//   make -C tests/standalone probe_implicit_tangent

#include <odelia/implicit_node.hpp>
#include <odelia/tangent.hpp>

#include <cmath>
#include <cstdio>

using odelia::ode::active_scalar;
using odelia::ode::adjoint_tape;
using odelia::ode::tangent_scalar;
using odelia::ode::seed_direction;
using odelia::ode::derivative_along;

namespace {

// The one active input this residual reads.
template <class T>
struct cube_inputs {
  using value_type = T;
  T p;
  template <class U>
  cube_inputs<U> rebind_from() const {
    return {U(odelia::util::to_passive(p))};
  }
};

// Written once, evaluated at two scalars.
template <class T>
T cube_residual(const T& y, const cube_inputs<T>& in) {
  return y * y * y - in.p;
}

template <class S>
S cube_root_at(double y_star, const S& p) {
  return odelia::implicit_value<S>(
      y_star, cube_inputs<S>{p},
      []<class T>(const T& y, const cube_inputs<T>& in) -> T {
        return cube_residual<T>(y, in);
      });
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

  std::printf("implicit_value: one residual, denominator on a tangent\n\n");
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

  std::printf("\n  %s\n", failures == 0 ? "all ok" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
