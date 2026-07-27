/* THE WITNESS for a separable_field whose source weights are implicit_value outputs.
 *
 * Everything measured so far assembles the field from closed-form source weights --
 * that is what K93 and FF16 do. TF24 does not: its source weight carries a leaf solve,
 * so the field's cumulative sums would be taken over quantities whose derivatives come
 * from the implicit function theorem rather than from a recorded expression. That
 * composition had no witness anywhere, in odelia or in plant, which is the gap this
 * file closes.
 *
 * Compiled on demand by test-ad-field-over-implicit.R (sourceCpp).
 *
 * The miniature keeps TF24's shape and none of its size:
 *
 *   sources        N individuals at heights H_j, descending (the field's own order)
 *   soil           ONE shared scalar theta every source reads -- the coupling, so a
 *                  single parameter reaches every cumulative sum
 *   leaf solve     per source, u_j solves F(u) = u(1+u) - k*psi(theta)*H_j = 0, with
 *                  psi(theta) = theta^-n a retention-shaped curve. dF/du = 1+2u > 0,
 *                  so denom_sign::positive is declared and checked
 *   source weight  amp * u_j * {1, H_j^-eta, H_j^-2eta}      <-- the IFT output is HERE
 *   query factors  {1, -2 z^eta, z^2eta} and their z-derivatives
 *   functional     J = sum_i [ exp(-A(z_i)) + w * dA/dz(z_i)^2 ]
 *
 * The rank-3 pair is the one from canopy_shape.h that all three strategies share, and
 * the functional reads BOTH `at` and `slope`, because the field's whole justification
 * over a spline is that one construct carries the value and the tangent.
 *
 * u* has a closed form, (-1 + sqrt(1 + 4*drive))/2, so this checks against an ANALYTIC
 * gradient and not only an FD -- an FD alone cannot distinguish a correct IFT partial
 * from one that is merely smooth.
 *
 * `sever_uptake` is the control. It freezes u_j with to_passive before the weight is
 * formed, which is exactly the severance the a1-a4 fixes removed elsewhere. Under it
 * dJ/dk and dJ/dtheta must collapse to zero, because k and theta reach J ONLY through
 * the leaf solve. A witness that cannot tell the severed case from the live one proves
 * nothing, so the test asserts both directions.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/implicit_node.hpp>
#include <odelia/ode_util.hpp>
#include <odelia/separable_field.hpp>

#include <array>
#include <cmath>
#include <vector>

using namespace Rcpp;

namespace {

constexpr std::size_t RANK = 3;

// Source heights, descending -- the order separable_field documents as the caller's
// duty. Plain double: a height is a coordinate, not a differentiated quantity here.
std::vector<double> heights(std::size_t n) {
  std::vector<double> h(n);
  for (std::size_t j = 0; j < n; ++j) h[j] = 20.0 * std::pow(0.93, static_cast<double>(j));
  return h;
}

// The leaf-shaped inner solve, off tape: u(1+u) = drive has the positive root
// (-1 + sqrt(1+4 drive))/2. Off-tape and exact, so implicit_value returns y* itself
// and the only thing on the tape is the IFT partial.
double solve_uptake(double drive) {
  return 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * drive));
}

// J(amp, eta, k, theta, n) for any scalar S. One body serves the double reference
// path, the FD, and the reverse tape -- so the comparison cannot drift between them.
template <class S>
S field_functional(const S& amp, const S& eta, const S& k, const S& theta, const S& n,
                   std::size_t n_src, double w, bool sever_uptake) {
  using std::exp;
  using std::pow;
  using std::sqrt;

  const std::vector<double> H = heights(n_src);

  // The retention curve: ONE shared scalar, read by every source. This is what makes
  // theta a genuine coupling parameter rather than a per-source constant.
  const S psi = pow(theta, -n);

  std::array<std::vector<S>, RANK> source_weight;
  for (std::size_t p = 0; p < RANK; ++p) source_weight[p].resize(n_src);

  for (std::size_t j = 0; j < n_src; ++j) {
    // drive is active in k, theta and n; the solve is done at its passive value.
    const S drive = k * psi * S(H[j]);
    const double drive_d = odelia::util::to_passive(drive);
    const double u_star = solve_uptake(drive_d);

    // The residual must return S exactly -- a deduced return type here would be an
    // XAD expression template holding references to the temporaries below.
    S u = odelia::implicit_value<S>(
        u_star, [&](S uu) -> S { return uu * (S(1.0) + uu) - drive; },
        odelia::denom_sign::positive);
    if (sever_uptake) {
      // The control: the value survives, the derivative does not.
      u = S(odelia::util::to_passive(u));
    }

    // The rank-3 source side, carrying the solve's output into every cumulative sum.
    const S hj = S(H[j]);
    const S base = amp * u;
    source_weight[0][j] = base;
    source_weight[1][j] = base * pow(hj, -eta);
    source_weight[2][j] = base * pow(hj, -2.0 * eta);
  }

  odelia::separable_field<S, RANK> field;
  field.assemble(source_weight);

  S total(0.0);
  for (std::size_t i = 0; i < n_src; ++i) {
    const S z = S(H[i]);
    const std::array<S, RANK> a = {S(1.0), S(-2.0) * pow(z, eta), pow(z, 2.0 * eta)};
    const std::array<S, RANK> a_prime = {
        S(0.0), S(-2.0) * eta * pow(z, eta - S(1.0)),
        S(2.0) * eta * pow(z, S(2.0) * eta - S(1.0))};
    const S A = field.at(a, i);
    const S dAdz = field.slope(a_prime, i);
    total += exp(-A) + S(w) * dAdz * dAdz;
  }
  return total;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List field_over_implicit_demo(double amp = 0.4, double eta = 2.0, double k = 0.8,
                                    double theta = 0.25, double n_psi = 1.5,
                                    int n_src = 12, double w = 0.05,
                                    bool sever_uptake = false, double fd_step = 1e-6) {
  using AD = xad::adj<double>::active_type;
  const std::size_t N = static_cast<std::size_t>(n_src);
  const std::vector<double> p0 = {amp, eta, k, theta, n_psi};

  // Reverse gradient.
  xad::adj<double>::tape_type tape;
  std::vector<AD> P(p0.begin(), p0.end());
  for (auto& v : P) tape.registerInput(v);
  tape.newRecording();
  AD J = field_functional<AD>(P[0], P[1], P[2], P[3], P[4], N, w, sever_uptake);
  tape.registerOutput(J);
  xad::derivative(J) = 1.0;
  tape.computeAdjoints();
  std::vector<double> grad(P.size());
  for (std::size_t i = 0; i < P.size(); ++i) grad[i] = xad::derivative(P[i]);

  // Central FD on the same body, in pure double.
  auto eval = [&](std::vector<double> q) -> double {
    return field_functional<double>(q[0], q[1], q[2], q[3], q[4], N, w, sever_uptake);
  };
  std::vector<double> fd(p0.size());
  for (std::size_t i = 0; i < p0.size(); ++i) {
    std::vector<double> lo = p0, hi = p0;
    const double h = fd_step * std::max(1.0, std::abs(p0[i]));
    lo[i] -= h;
    hi[i] += h;
    fd[i] = (eval(hi) - eval(lo)) / (2.0 * h);
  }

  return Rcpp::List::create(
      Rcpp::Named("value") = xad::value(J),
      Rcpp::Named("value_double") = eval(p0),
      Rcpp::Named("names") =
          CharacterVector::create("amp", "eta", "k", "theta", "n_psi"),
      Rcpp::Named("grad") = wrap(grad),
      Rcpp::Named("fd") = wrap(fd),
      Rcpp::Named("severed") = sever_uptake);
}

/* The analytic reference for the one channel an FD cannot vouch for.
 *
 * amp scales every source weight linearly, so it factors straight out of both
 * cumulative sums: A(z) is homogeneous of degree 1 in amp ONLY IF u did not depend on
 * amp -- and it does not. So dA/damp = A/amp exactly, giving
 *     dJ/damp = sum_i [ -exp(-A_i) * A_i + 2 w * dAdz_i^2 ] / amp.
 * This is independent of the IFT machinery, so agreement pins the field assembly; the
 * k and theta channels then pin the IFT partials against the closed-form u*.
 */
// [[Rcpp::export]]
Rcpp::List field_over_implicit_amp_identity(double amp = 0.4, double eta = 2.0,
                                            double k = 0.8, double theta = 0.25,
                                            double n_psi = 1.5, int n_src = 12,
                                            double w = 0.05) {
  const std::size_t N = static_cast<std::size_t>(n_src);
  const std::vector<double> H = heights(N);
  const double psi = std::pow(theta, -n_psi);

  std::array<std::vector<double>, RANK> sw;
  for (std::size_t p = 0; p < RANK; ++p) sw[p].resize(N);
  for (std::size_t j = 0; j < N; ++j) {
    const double u = solve_uptake(k * psi * H[j]);
    const double base = amp * u;
    sw[0][j] = base;
    sw[1][j] = base * std::pow(H[j], -eta);
    sw[2][j] = base * std::pow(H[j], -2.0 * eta);
  }
  odelia::separable_field<double, RANK> field;
  field.assemble(sw);

  double dJ_damp = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double z = H[i];
    const std::array<double, RANK> a = {1.0, -2.0 * std::pow(z, eta),
                                        std::pow(z, 2.0 * eta)};
    const std::array<double, RANK> ap = {
        0.0, -2.0 * eta * std::pow(z, eta - 1.0),
        2.0 * eta * std::pow(z, 2.0 * eta - 1.0)};
    const double A = field.at(a, i);
    const double dAdz = field.slope(ap, i);
    dJ_damp += (-std::exp(-A) * A + 2.0 * w * dAdz * dAdz) / amp;
  }
  return Rcpp::List::create(Rcpp::Named("dJ_damp_analytic") = dJ_damp);
}
