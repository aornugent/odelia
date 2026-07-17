// Generic MRI-GARK macro step in component-partition (solve-decoupled, explicit-slow)
// form, following the algorithm in the MRI brief §3. Scalar type double here; the
// odelia port templates on S. Coupling "lives in data" (a MRIStepCoupling-shaped
// table); the fast block is a black-box IVP; the slow block follows its exact
// within-leg polynomial path and is never sub-stepped.
#ifndef MRI_CORE_HPP
#define MRI_CORE_HPP
#include <vector>
#include <functional>
#include <cmath>
#include <cstdio>

// Coupling table: s slow stages (F_1..F_s), nmat = K+1 coupling matrices.
// c has s+1 abscissae (c[0]=0 ... c[s]=1). G[k] is (s+1) x s (rows i=0..s, cols j=0..s-1).
struct MRICoupling {
  int s = 0, K = 0, q = 0;
  std::vector<double> c;                                   // size s+1
  std::vector<std::vector<std::vector<double>>> G;         // [K+1][s+1][s]
  const char* name = "";
  double abar(int i, int j) const {                       // ā_ij = Σ_k G[k]_ij/(k+1)
    double a = 0.0; for (int k = 0; k <= K; ++k) a += G[k][i][j] / (k + 1); return a;
  }
};

// Lift an explicit base Butcher table (A: s×s, b: s, c: s) to a K=0 MRI by the
// row-difference rule ā_ij = A^cum_ij − A^cum_{i-1,j} with A^cum = base A, last row = b.
// By construction the induced base ERK equals (A,b,c): collapse identity 1 holds.
inline MRICoupling mis_lift(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& b,
                            const std::vector<double>& cbase,
                            int order, const char* name) {
  int s = (int)b.size();
  MRICoupling M; M.s = s; M.K = 0; M.q = order; M.name = name;
  M.c.resize(s + 1);
  for (int i = 0; i < s; ++i) M.c[i] = cbase[i];
  M.c[s] = 1.0;
  M.G.assign(1, std::vector<std::vector<double>>(s + 1, std::vector<double>(s, 0.0)));
  // cumulative rows: row i (0..s-1) = A[i][:]; row s = b[:]
  auto cumrow = [&](int i, int j) -> double { return (i < s) ? A[i][j] : b[j]; };
  for (int i = 0; i <= s; ++i)
    for (int j = 0; j < s; ++j) {
      double hi = cumrow(i, j);
      double lo = (i == 0) ? 0.0 : cumrow(i - 1, j);
      M.G[0][i][j] = hi - lo;                              // ā_ij (K=0)
    }
  return M;
}

// ---- built-in tables (base ERKs lifted to K=0 MIS) ------------------------
inline MRICoupling table_fwd_euler() {                    // order 1
  return mis_lift({{0.0}}, {1.0}, {0.0}, 1, "MRI-FwdEuler");
}
inline MRICoupling table_midpoint() {                     // explicit midpoint, order 2
  return mis_lift({{0.0,0.0},{0.5,0.0}}, {0.0,1.0}, {0.0,0.5}, 2, "MRI-Midpoint2");
}
inline MRICoupling table_heun() {                         // Heun (trapezoidal), order 2
  return mis_lift({{0.0,0.0},{1.0,0.0}}, {0.5,0.5}, {0.0,1.0}, 2, "MRI-Heun2");
}
inline MRICoupling table_kutta3() {                       // Kutta's third-order rule
  return mis_lift({{0.0,0.0,0.0},{0.5,0.0,0.0},{-1.0,2.0,0.0}},
                  {1.0/6,2.0/3,1.0/6}, {0.0,0.5,1.0}, 3, "MRI-Kutta3");
}

// Row-sum / consistency check: Σ_j ā_ij = Δc_i, and Σ_i Σ_j ā_ij = 1.
inline bool check_rowsums(const MRICoupling& M, double tol = 1e-13) {
  bool ok = true;
  std::vector<double> bsum(M.s, 0.0);
  for (int i = 1; i <= M.s; ++i) {
    double rs = 0.0; for (int j = 0; j < M.s; ++j) { rs += M.abar(i, j); bsum[j] += M.abar(i, j); }
    // add row 0 contribution (usually zero)
    if (i == 1) for (int j = 0; j < M.s; ++j) bsum[j] += M.abar(0, j);
    double dc = M.c[i] - M.c[i - 1];
    if (std::fabs(rs - dc) > tol) { ok = false; printf("  rowsum fail i=%d: %.3e vs Δc=%.3e\n", i, rs, dc); }
  }
  return ok;
}

// One MRI macro step, component partition.
//   x (slow, Nx), u (fast, Nu).
//   slow(x,u) -> dx           : slow tendency vector (reads fast state)
//   agg(vec)  -> double        : the linear slow-aggregate the fast block reads (e.g. mean)
//   inner(u, poly, ta, tb)     : advance u over [ta,tb] with the slow aggregate following
//                                x̄(τ) = Σ_m poly[m] τ^m, τ=(t-ta)/(tb-ta), solved to tolerance.
// The fast IVP carries NO explicit forcing (component partition): the slow influence
// enters only through the aggregate polynomial the inner reads.
template <class Slow, class Agg, class Inner>
void mri_macro_step(const MRICoupling& M, double t, double H,
                    std::vector<double>& x, std::vector<double>& u,
                    Slow slow, Agg agg, Inner inner) {
  int s = M.s, Nx = (int)x.size();
  std::vector<std::vector<double>> F(s + 1);              // F[j].x, j=1..s (1-indexed)
  std::vector<double> Fbar(s + 1, 0.0);                   // agg(F[j].x)
  // stage 1
  F[1] = slow(x, u); Fbar[1] = agg(F[1]);
  std::vector<double> zx = x, zu = u;
  for (int i = 2; i <= s + 1; ++i) {
    double dc = M.c[i - 1] - M.c[i - 2];                  // Δc_i  (c is 0-indexed: c[i-1])
    if (dc <= 0.0) {                                      // instantaneous slow update
      for (int a = 0; a < Nx; ++a) {
        double inc = 0.0; for (int j = 1; j < i; ++j) inc += M.abar(i - 1, j - 1) * F[j][a];
        zx[a] += H * inc;
      }
    } else {
      // aggregate polynomial x̄(τ): x̄_start + H Σ_k [Σ_{j<i} G[k]_{i-1,j-1} F̄[j]] τ^{k+1}/(k+1)
      std::vector<double> poly(M.K + 2, 0.0);
      poly[0] = agg(zx);
      for (int k = 0; k <= M.K; ++k) {
        double sj = 0.0; for (int j = 1; j < i; ++j) sj += M.G[k][i - 1][j - 1] * Fbar[j];
        poly[k + 1] += H * sj / (k + 1);
      }
      double ta = t + M.c[i - 2] * H, tb = t + M.c[i - 1] * H;
      inner(zu, poly, ta, tb);                            // advance fast block (reads x̄(τ))
      // slow endpoint x_leg(1) = zx + H Σ_{j<i} ā_{i-1,j} F[j].x
      for (int a = 0; a < Nx; ++a) {
        double inc = 0.0; for (int j = 1; j < i; ++j) inc += M.abar(i - 1, j - 1) * F[j][a];
        zx[a] += H * inc;
      }
    }
    if (i <= s) { F[i] = slow(zx, zu); Fbar[i] = agg(F[i]); }  // slow reads fast at abscissa c_i
  }
  x = zx; u = zu;
}

#endif
