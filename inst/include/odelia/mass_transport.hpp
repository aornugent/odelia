#ifndef ODELIA_MASS_TRANSPORT_HPP_
#define ODELIA_MASS_TRANSPORT_HPP_

#include <cmath>
#include <cstddef>
#include <vector>

namespace odelia {

// The neighbour spacing of cohorts held in descending size order
// (x_0 > x_1 > ... > x_{N-1}): centred, (x_{i-1} - x_{i+1}) / 2, and one-sided
// at the ends. Positive throughout. This is the weight every population
// reduction uses; log_density_rate reuses the same neighbour differences, so
// the two are one discrete operator.
template <class S>
std::vector<S> cohort_spacing(const std::vector<S>& x) {
  const std::size_t n = x.size();
  std::vector<S> dx(n);
  if (n == 1) { dx[0] = S(0.0); return dx; }
  dx[0]     = x[0] - x[1];
  dx[n - 1] = x[n - 2] - x[n - 1];
  for (std::size_t i = 1; i + 1 < n; ++i) dx[i] = (x[i - 1] - x[i + 1]) / 2.0;
  return dx;
}

// The transport-log-mass chart. The population state is carried as log mass
//
//     lambda_i = log_density_i + log(cohort_spacing_i),
//
// which evolves by d(lambda)/dt = -loss (the compression C cancels identically,
// so it is never formed -- no numerical d(g)/d(size), stable through a growth
// stall, and no severed derivative on the tape). log density is a read-side view
// reconstructed from lambda and the SAME cohort_spacing that defines the chart;
// sharing the one operator is what keeps the value and its parameter-derivative
// self-consistent. These two are inverse maps on that shared spacing:
//
//   log_mass_from_log_density : seed lambda from a freshly set birth log-density.
//   log_density_from_log_mass : the view a model reads to express in log-density.
template <class S>
std::vector<S> log_mass_from_log_density(const std::vector<S>& x,
                                         const std::vector<S>& log_density) {
  using std::log;
  const std::vector<S> dx = cohort_spacing(x);
  std::vector<S> lm(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) lm[i] = log_density[i] + log(dx[i]);
  return lm;
}
template <class S>
std::vector<S> log_density_from_log_mass(const std::vector<S>& x,
                                         const std::vector<S>& log_mass) {
  using std::log;
  const std::vector<S> dx = cohort_spacing(x);
  std::vector<S> ld(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) ld[i] = log_mass[i] - log(dx[i]);
  return ld;
}

// The density-transport rate: d(log density)/dt = -C - loss, with C the
// discrete d(g)/d(size) formed as the spacing operator applied to the growth
// rate over the same operator applied to the coordinate,
//
//     C_i = cohort_spacing(g)_i / cohort_spacing(x)_i   (0 for a lone cohort).
//
// Defining C through cohort_spacing is the point: C is then, by construction,
// the log-rate of the spacing's own evolution, so transported as log mass (log
// density + log spacing) the two cancel and the rate is just -loss. Sharing the
// one operator makes that cancellation hold in the parameter derivative, not
// only the value -- so the compression leaves no residual in a conserved
// moment, and the two cannot silently drift apart.
template <class S>
std::vector<S> log_density_rate(const std::vector<S>& x, const std::vector<S>& g,
                                const std::vector<S>& loss) {
  const std::size_t n = x.size();
  std::vector<S> rate(n);
  if (n == 1) { rate[0] = -loss[0]; return rate; }
  const std::vector<S> dg = cohort_spacing(g);
  const std::vector<S> dx = cohort_spacing(x);
  for (std::size_t i = 0; i < n; ++i) rate[i] = -dg[i] / dx[i] - loss[i];
  return rate;
}

}  // namespace odelia

#endif
