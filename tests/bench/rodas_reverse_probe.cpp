// Feasibility probe for reverse-mode RODAS (#36).
//
// A RODAS step solves W k_i = rhs_i with W = I/(h*gamma) - J, J = df/dy. So the
// step output depends on the parameters both directly (through the RHS) and
// through J. A correct reverse-mode gradient must therefore capture dJ/d(param).
// This probe checks whether the *existing* machinery (run the step under the
// adjoint scalar and tape it, as compute_jacobian does for RKCK) can deliver that.
//
// TWO INDEPENDENT BLOCKERS were found; either is fatal to the naive approach.
//
// BLOCKER 1 (type level -- does not compile). Taping the Jacobian's forward sweep
// under an adjoint value_type needs the tangent type FReal<AReal<double>>. XAD's
// vendored build cannot construct FReal<AReal<double>> from an AReal<double>:
// FReal<Scalar>::nested_type is the *bottom* float (double), and its value ctor
// takes that double, with no AReal->double conversion. ode_jacobian.hpp:94
// (`v[j] = tangent_type(y[j])`) fails to compile with:
//     error: no matching function for call to
//       'xad::FReal<xad::AReal<double,1>,1>::FReal(const xad::AReal<double,1>&)'
// This is the "higher-order AD (Scalar != nested_type) is not supported"
// limitation noted in XAD/Literals.hpp. So Jacobian<System> cannot be
// instantiated under an adjoint scalar at all -- the forward-over-adjoint
// nesting the naive path assumes does not exist in this build.
//
// BLOCKER 2 (semantic -- compiles, but wrong). Even setting the Jacobian aside,
// rebind_from() lifts a System to another scalar by *value only* (xad::value),
// so a twin built from an active system is disconnected from the taped
// parameters. Any derivative taken through a rebind_from twin drops the
// parameter dependence. Probe B below demonstrates this at first order (where it
// does compile): d/dtheta taped through a rebind_from twin is identically zero.
//
// Conclusion: reverse-mode RODAS is NOT reachable by taping the existing step.
// It needs the discrete adjoint of the step (transposed solves reusing W, plus
// Hessian-vector products of f for the dJ terms) injected via a CheckpointCallback
// -- never taping the linear algebra. See the write-up for the worked design.

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <functional>
#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include <odelia/ode_interface.hpp>

using namespace odelia;

// f(y) = -theta * y^2.  d f / d theta = -y^2 (nonzero), so a correct reverse
// pass through anything that rebuilds the System must preserve it.
template <typename T = double>
class Sys1 {
public:
  using value_type = T;
  explicit Sys1(T theta_) : theta(theta_), y0(T(0)), t0(0.0), time(0.0) {}

  size_t ode_size() const { return 1; }
  double ode_t0() const { return t0; }
  double ode_time() const { return time; }

  template <typename It>
  It set_ode_state(It it, double time_) { time = time_; y0 = *it++; return it; }
  template <typename It>
  It ode_rates(It it) const { *it++ = -theta * y0 * y0; return it; }

  template <class U> using rebind = Sys1<U>;
  template <class U>
  rebind<U> rebind_from() const { return Sys1<U>(U(xad::value(theta))); }

private:
  T theta, y0;
  double t0, time;
};

// [[Rcpp::export]]
Rcpp::List rodas_reverse_probe(double theta = 1.7, double y = 0.9) {
  using ad = xad::adj<double>;
  using ad_type = ad::active_type;   // AReal<double>
  const double t = 0.0;

  auto run = [](std::vector<ad_type>& in,
                std::function<std::vector<ad_type>(std::vector<ad_type>&)>& f) {
    ad::tape_type tp(false);
    tp.activate();
    const double r = xad::computeJacobian(in, f, 1U, &tp)[0][0];
    tp.deactivate();
    return r;
  };

  // Probe A (control): d(f)/dtheta taped on the active system directly. The
  // parameter is live on the tape -> first-order reverse mode works.
  std::vector<ad_type> inA{ad_type(theta)};
  std::function<std::vector<ad_type>(std::vector<ad_type>&)> fA =
      [&](std::vector<ad_type>& x) {
        Sys1<ad_type> sys(x[0]);
        std::vector<ad_type> yv{ad_type(y)}, d(1);
        ode::derivs(sys, yv, d, t);
        return d;
      };
  const double dfdtheta_direct = run(inA, fA);

  // Probe B (the blocker): same derivative, but taken through a rebind_from twin
  // -- exactly how Jacobian::compute obtains the System it differentiates. The
  // twin is built by value, so theta is severed from the tape.
  std::vector<ad_type> inB{ad_type(theta)};
  std::function<std::vector<ad_type>(std::vector<ad_type>&)> fB =
      [&](std::vector<ad_type>& x) {
        Sys1<ad_type> live(x[0]);
        auto twin = live.rebind_from<ad_type>();   // value-only lift
        std::vector<ad_type> yv{ad_type(y)}, d(1);
        ode::derivs(twin, yv, d, t);
        return d;
      };
  const double dfdtheta_via_twin = run(inB, fB);

  return Rcpp::List::create(
      Rcpp::Named("dfdtheta_true") = -y * y,
      Rcpp::Named("dfdtheta_direct") = dfdtheta_direct,
      Rcpp::Named("dfdtheta_via_rebind_twin") = dfdtheta_via_twin);
}
