/* Exercises odelia::separable_field on the shading kernel
 *     kappa(z, x) = m(x) * (1 - (z/x)^eta)^2,   m(x) = (pi/4) x^2,
 * an exact rank-3 separable sum, compiled on demand by test-ad-separable-field.R (sourceCpp).
 *
 * A population of cohorts in descending size order casts one-sided shade; the
 * field at a query is A(z) = sum_{x_j >= z} kappa(z, x_j) m_j. The demo returns,
 * for a fixed population, three checks the test asserts:
 *   - separability: sum_p a_p(z) b_p(x) == kappa(z, x);
 *   - the assembled field and its query slope vs the direct O(N^2) sum;
 *   - the FD-free dot-product oracle  J v == <v, gradient>  (forward-mode tangent
 *     through the field vs the reverse adjoint), over the cohort sizes as inputs.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/separable_field.hpp>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace Rcpp;

static constexpr std::size_t kRank = 3;
static const double kPi = 3.14159265358979323846;

template <class S> static S mfun(const S& x) { return (kPi / 4.0) * x * x; }

template <class S> static std::array<S, kRank> a_of(const S& z, double eta) {
  using std::pow;
  S ze = pow(z, eta);
  return {S(1.0), S(-2.0) * ze, ze * ze};
}
template <class S> static std::array<S, kRank> aprime_of(const S& z, double eta) {
  using std::pow;
  return {S(0.0), S(-2.0 * eta) * pow(z, eta - 1.0),
          S(2.0 * eta) * pow(z, 2.0 * eta - 1.0)};
}
template <class S> static std::array<S, kRank> b_of(const S& x, double eta) {
  using std::pow;
  S m = mfun(x), xe = pow(x, eta);
  return {m, m / xe, m / (xe * xe)};
}
template <class S> static S kappa_direct(const S& z, const S& x, double eta) {
  using std::pow;
  S u = z / x, q = S(1.0) - pow(u, eta);
  return mfun(x) * q * q;
}
template <class S> static S kappa_z_direct(const S& z, const S& x, double eta) {
  using std::pow;
  S u = z / x, q = S(1.0) - pow(u, eta);
  return mfun(x) * S(2.0) * q * (S(-eta) * pow(u, eta - 1.0) / x);
}

// Point read at each cohort's own height: sum_i w_i * A(x_i). Scalar-generic so
// the reverse tape and the forward tangent run the identical assembly.
template <class S>
static S reduce(const std::vector<S>& x, const std::vector<double>& m,
                const std::vector<double>& w, double eta) {
  const std::size_t n = x.size();
  std::array<std::vector<S>, kRank> sw;
  for (std::size_t p = 0; p < kRank; ++p) sw[p].resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    auto b = b_of(x[j], eta);
    for (std::size_t p = 0; p < kRank; ++p) sw[p][j] = b[p] * m[j];
  }
  odelia::separable_field<S, kRank> field;
  field.assemble(sw);
  S total(0.0);
  for (std::size_t i = 0; i < n; ++i) total += w[i] * field.at(a_of(x[i], eta), i);
  return total;
}

// [[Rcpp::export]]
Rcpp::List separable_field_demo(double eta = 4.0, int n = 8) {
  std::vector<double> x(n), m(n), w(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 5.0 - 0.4 * i;              // descending sizes
    m[i] = 0.3 + 0.1 * i;
    w[i] = 0.7 + 0.05 * i;
  }

  // Separability at several (z, x >= z).
  std::vector<double> sep_recomb, sep_direct;
  for (double z : {2.5, 3.1, 4.0}) {
    const double xx = 4.5;
    auto a = a_of(z, eta);
    auto b = b_of(xx, eta);
    double s = 0.0;
    for (std::size_t p = 0; p < kRank; ++p) s += a[p] * b[p];
    sep_recomb.push_back(s);
    sep_direct.push_back(kappa_direct(z, xx, eta));
  }

  // Field and slope vs the direct sum over taller-or-equal sources.
  std::array<std::vector<double>, kRank> sw;
  for (std::size_t p = 0; p < kRank; ++p) sw[p].resize(n);
  for (int j = 0; j < n; ++j) {
    auto b = b_of(x[j], eta);
    for (std::size_t p = 0; p < kRank; ++p) sw[p][j] = b[p] * m[j];
  }
  odelia::separable_field<double, kRank> field;
  field.assemble(sw);
  std::vector<double> field_assembled, field_direct, slope_assembled, slope_direct;
  for (int i = 0; i < n; ++i) {
    double A = 0.0, Sd = 0.0;
    for (int j = 0; j <= i; ++j) {
      A  += kappa_direct(x[i], x[j], eta) * m[j];
      Sd += kappa_z_direct(x[i], x[j], eta) * m[j];
    }
    field_assembled.push_back(field.at(a_of(x[i], eta), i));
    field_direct.push_back(A);
    slope_assembled.push_back(field.slope(aprime_of(x[i], eta), i));
    slope_direct.push_back(Sd);
  }

  // Dot-product oracle over the cohort sizes as differentiable inputs.
  using ad = xad::adj<double>;
  using AD = ad::active_type;
  ad::tape_type tape;
  std::vector<AD> xa(x.begin(), x.end());
  for (auto& v : xa) tape.registerInput(v);
  tape.newRecording();
  AD F = reduce<AD>(xa, m, w, eta);
  tape.registerOutput(F);
  xad::derivative(F) = 1.0;
  tape.computeAdjoints();
  std::vector<double> grad(n);
  for (int j = 0; j < n; ++j) grad[j] = xad::derivative(xa[j]);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::vector<double> dir(n);
  for (auto& d : dir) d = unif(rng);

  using FAD = xad::fwd<double>::active_type;
  std::vector<FAD> xf(x.begin(), x.end());
  for (int j = 0; j < n; ++j) xad::derivative(xf[j]) = dir[j];
  FAD Ff = reduce<FAD>(xf, m, w, eta);

  double dot = 0.0;
  for (int j = 0; j < n; ++j) dot += grad[j] * dir[j];

  return Rcpp::List::create(
      Rcpp::Named("sep_recomb") = wrap(sep_recomb),
      Rcpp::Named("sep_direct") = wrap(sep_direct),
      Rcpp::Named("field_assembled") = wrap(field_assembled),
      Rcpp::Named("field_direct") = wrap(field_direct),
      Rcpp::Named("slope_assembled") = wrap(slope_assembled),
      Rcpp::Named("slope_direct") = wrap(slope_direct),
      Rcpp::Named("value_fwd") = xad::value(Ff),
      Rcpp::Named("value_rev") = xad::value(F),
      Rcpp::Named("jvp") = xad::derivative(Ff),
      Rcpp::Named("dot_v_grad") = dot);
}
