/* Exercises odelia mass transport -- cohort_spacing and log_density_rate --
 * compiled on demand by test-ad-mass-transport.R (sourceCpp).
 *
 * The density-transport rate is d(log density)/dt = -C - loss, C the neighbour
 * secant of the growth rate. Because C reuses the spacing every reduction
 * weights with, transported as log mass the compression cancels and the rate is
 * just -loss. The demo returns three checks the test asserts:
 *   - cancellation: -rate == the log-rate of the spacing's own evolution, so the
 *     compression leaves no residual (value and, via the oracle, derivative);
 *   - the neighbour secant against the analytic d(g)/dx;
 *   - the dot-product oracle  J v == <v, gradient>  over the cohort positions.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/mass_transport.hpp>

#include <random>
#include <vector>

using namespace Rcpp;

template <class S> static S gfun(const S& x) { return S(3.0) - 0.2 * x - 0.05 * x * x; }
template <class S> static S gprime(const S& x) { return S(-0.2) - 0.1 * x; }

// Scalar reduction of the transport rate over the population, generic on S so
// the reverse tape and the forward tangent run the identical code.
template <class S>
static S reduce(const std::vector<S>& x, const std::vector<double>& w,
                const std::vector<double>& loss_d) {
  std::vector<S> g(x.size()), loss(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) { g[i] = gfun(x[i]); loss[i] = S(loss_d[i]); }
  auto rate = odelia::log_density_rate(x, g, loss);
  S total(0.0);
  for (std::size_t i = 0; i < x.size(); ++i) total += w[i] * rate[i];
  return total;
}

// [[Rcpp::export]]
Rcpp::List mass_transport_demo(int n = 8) {
  std::vector<double> x(n), w(n), zero(n, 0.0), loss(n), g(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 5.0 - 0.4 * i;                 // descending
    w[i] = 0.7 + 0.05 * i;
    loss[i] = 0.1 + 0.02 * i;
  }
  for (int i = 0; i < n; ++i) g[i] = gfun(x[i]);

  // Cancellation: -rate == d(dx)/dt / dx (spacing operator on g over spacing on x).
  auto dx = odelia::cohort_spacing(x);
  auto ddx = odelia::cohort_spacing(g);
  auto rate0 = odelia::log_density_rate(x, g, zero);
  std::vector<double> cancel_rate(n), cancel_ref(n);
  for (int i = 0; i < n; ++i) { cancel_rate[i] = -rate0[i]; cancel_ref[i] = ddx[i] / dx[i]; }

  // Neighbour secant (interior) vs analytic dg/dx.
  std::vector<double> secant, gprime_ref;
  for (int i = 1; i + 1 < n; ++i) { secant.push_back(-rate0[i]); gprime_ref.push_back(gprime(x[i])); }

  // Dot-product oracle over the cohort positions.
  using ad = xad::adj<double>;
  using AD = ad::active_type;
  ad::tape_type tape;
  std::vector<AD> xa(x.begin(), x.end());
  for (auto& v : xa) tape.registerInput(v);
  tape.newRecording();
  AD F = reduce<AD>(xa, w, loss);
  tape.registerOutput(F);
  xad::derivative(F) = 1.0;
  tape.computeAdjoints();
  std::vector<double> grad(n);
  for (int j = 0; j < n; ++j) grad[j] = xad::derivative(xa[j]);

  std::mt19937 rng(999);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::vector<double> dir(n);
  for (auto& d : dir) d = unif(rng);

  using FAD = xad::fwd<double>::active_type;
  std::vector<FAD> xf(x.begin(), x.end());
  for (int j = 0; j < n; ++j) xad::derivative(xf[j]) = dir[j];
  FAD Ff = reduce<FAD>(xf, w, loss);

  double dot = 0.0;
  for (int j = 0; j < n; ++j) dot += grad[j] * dir[j];

  return Rcpp::List::create(
      Rcpp::Named("cancel_rate") = wrap(cancel_rate),
      Rcpp::Named("cancel_ref") = wrap(cancel_ref),
      Rcpp::Named("secant") = wrap(secant),
      Rcpp::Named("gprime_ref") = wrap(gprime_ref),
      Rcpp::Named("value_fwd") = xad::value(Ff),
      Rcpp::Named("value_rev") = xad::value(F),
      Rcpp::Named("jvp") = xad::derivative(Ff),
      Rcpp::Named("dot_v_grad") = dot);
}
