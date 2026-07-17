#ifndef ODELIA_SEPARABLE_FIELD_HPP_
#define ODELIA_SEPARABLE_FIELD_HPP_

#include <array>
#include <cstddef>
#include <vector>

namespace odelia {

// The one-sided field a population casts along a scalar coordinate when the
// kernel splits into a finite sum of factors, one side depending on the query
// z and the other on the source x:
//
//     kappa(z, x) = sum_p a_p(z) * b_p(x)   for x >= z, else 0.
//
// Separability is what makes the field cheap and exact: the aggregate at a
// query becomes a weighting of R cumulative sums over the population, so it
// needs no reconstruction, no fitted knots, and no recorded positions.
//
//     A(z)     = sum_{x_j >= z} kappa(z, x_j) * m_j = sum_p a_p(z) * B_p(z),
//     dA/dz(z) = sum_p a_p'(z) * B_p(z),
//     B_p(z)   = sum_{x_j >= z} b_p(x_j) * m_j.
//
// The sources are held in descending size order (x_0 > x_1 > ... > x_{N-1}), so
// "x_j >= z" for a query ranked at i is the running total of ranks 0..i -- one
// cumulative sum per factor. The caller supplies that order; the field sees
// only the per-source weights b_p(x_j) * m_j, not the sizes. Rank i's own
// contribution carries the diagonal value kappa(x_i, x_i), which is a double
// zero for the shading kernels, so including or excluding the self term reads
// the same field -- the cumulative is taken inclusive.
//
// Generic on S so the same assembly runs on double, on a forward tangent, and
// on the reverse tape.
template <class S, std::size_t R>
class separable_field {
public:
  // source_weight[p][j] = b_p(x_j) * m_j, source j in descending size order.
  void assemble(const std::array<std::vector<S>, R>& source_weight) {
    n_ = source_weight[0].size();
    for (std::size_t p = 0; p < R; ++p) {
      cumulative_[p].resize(n_);
      S running(0.0);
      for (std::size_t j = 0; j < n_; ++j) {
        running += source_weight[p][j];
        cumulative_[p][j] = running;
      }
    }
  }

  std::size_t size() const { return n_; }

  // A(z) at the query ranked at i: every source of rank 0..i is at least as
  // large, so the field is the query factors a[p] weighting the cumulatives.
  S at(const std::array<S, R>& a, std::size_t i) const {
    S total(0.0);
    for (std::size_t p = 0; p < R; ++p) total += a[p] * cumulative_[p][i];
    return total;
  }

  // dA/dz at the same query, from the query-factor derivatives a_p'(z).
  S slope(const std::array<S, R>& a_prime, std::size_t i) const {
    S total(0.0);
    for (std::size_t p = 0; p < R; ++p) total += a_prime[p] * cumulative_[p][i];
    return total;
  }

private:
  std::size_t n_ = 0;
  std::array<std::vector<S>, R> cumulative_;
};

}  // namespace odelia

#endif
