// -*-c++-*-
#ifndef ODELIA_MRI_HPP_
#define ODELIA_MRI_HPP_

#include <vector>
#include <cstddef>
#include <odelia/ode_solver_internal.hpp>

// Multirate-infinitesimal (MRI-GARK) stepping for a System that splits into a
// slow block x and a fast block u. The outer method is an explicit Runge-Kutta
// on x; between its stages the fast block is sub-cycled by the existing adaptive
// solver over one leg, reading the slow influence as a low-dimensional signal
// g that follows a polynomial in the leg-local time. A System is multirate when,
// on top of the plain ODE interface, it provides:
//   size_t fast_size(), slow_size(), coupling_size();
//   void slow_rates(x, u, dxdt);   // slow tendency, reads the fast block
//   void fast_rates(u, g, dudt);   // fast tendency, reads the slow signal g
//   void aggregate(x, g);          // the signal the fast block reads (see below)
// aggregate must be linear and homogeneous (g = P*x, no constant term): the
// method applies it to both the slow state and its tendency.

namespace odelia {
namespace ode {

// MRI-GARK coupling (Sandu, SINUM 2019): `nodes` abscissae c[0..nodes-1] with
// c[0]=0, c[nodes-1]=1, and `nmat` coupling matrices G[k] (each nodes x nodes).
// The slow block advances by the induced RK increments abar(i,j) = sum_k
// G[k][i][j]/(k+1); the fast block reads a degree-nmat polynomial in the
// leg-local time built from the same G. nmat=1 is a multirate-infinitesimal-step
// method; nmat=2 carries the extra coupling of a genuine order-3 MRI-GARK.
struct MRICoupling {
  int nodes;
  int nmat;
  int order;
  std::vector<double> c;                               // [nodes]
  std::vector<std::vector<std::vector<double>>> G;     // [nmat][nodes][nodes]
  const char* name;

  double abar(int i, int j) const {
    double a = 0.0;
    for (int k = 0; k < nmat; ++k) a += G[k][i][j] / (k + 1);
    return a;
  }
};

// Lift an explicit base Butcher table (A, b, c) to a nmat=1 MRI coupling by the
// row-difference rule G[i][j] = A[i][j] - A[i-1][j], padding a final node so the
// last abscissa is 1 and the padded row is b - A[last]. The induced slow method
// is exactly (A, b, c), so with a zero fast block the macro step reproduces it.
inline MRICoupling mri_lift(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& b,
                            const std::vector<double>& c,
                            int order, const char* name) {
  const int s = (int)b.size();
  const double tol = 1e-14;
  bool pad = std::fabs(c[s - 1] - 1.0) > tol;
  for (int j = 0; j < s && !pad; ++j) pad = std::fabs(A[s - 1][j] - b[j]) > tol;
  const int n = pad ? s + 1 : s;

  MRICoupling M;
  M.nodes = n; M.nmat = 1; M.order = order; M.name = name;
  M.c.assign(n, 1.0);
  for (int i = 0; i < s; ++i) M.c[i] = c[i];
  M.G.assign(1, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));
  for (int i = 1; i < s; ++i)
    for (int j = 0; j < s; ++j) M.G[0][i][j] = A[i][j] - A[i - 1][j];
  if (pad)
    for (int j = 0; j < s; ++j) M.G[0][n - 1][j] = b[j] - A[s - 1][j];
  return M;
}

inline MRICoupling mri_forward_euler() {
  return mri_lift({{0.0}}, {1.0}, {0.0}, 1, "MRI-ForwardEuler");
}
inline MRICoupling mri_heun() {
  return mri_lift({{0.0, 0.0}, {1.0, 0.0}}, {0.5, 0.5}, {0.0, 1.0}, 2, "MRI-Heun");
}
inline MRICoupling mri_kutta3() {
  return mri_lift({{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {-1.0, 2.0, 0.0}},
                  {1.0 / 6, 2.0 / 3, 1.0 / 6}, {0.0, 0.5, 1.0}, 3, "MRI-Kutta3");
}

// MRI-GARK-ERK33a (Sandu, SINUM 57:2300-2327, 2019): order 3, nmat=2, the AD
// default in ARKODE's MRIStep. Coupling W[0], W[1] transcribed from that table;
// each interval row sums to its abscissa gap.
inline MRICoupling mri_erk33a() {
  MRICoupling M;
  M.nodes = 4; M.nmat = 2; M.order = 3; M.name = "MRI-GARK-ERK33a";
  M.c = {0.0, 1.0 / 3, 2.0 / 3, 1.0};
  M.G.assign(2, std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0)));
  M.G[0][1][0] =  1.0 / 3;
  M.G[0][2][0] = -1.0 / 3;   M.G[0][2][1] = 2.0 / 3;
  M.G[0][3][1] = -2.0 / 3;   M.G[0][3][2] = 1.0;
  M.G[1][3][0] =  1.0 / 2;   M.G[1][3][2] = -1.0 / 2;
  return M;
}

// The fast block presented as a plain System over a single leg: its rates read
// the slow signal g(theta), theta the leg-local time in [0,1], as the polynomial
// captured for the leg. This lets the existing adaptive solver sub-cycle it.
template <class System>
class FastLeg {
public:
  using value_type = typename System::value_type;
  using vec = std::vector<value_type>;

  FastLeg(const System& sys, const vec& u0,
          const std::vector<vec>& a_poly, double ta, double dt)
    : sys_(sys), a_poly_(a_poly), ta_(ta), dt_(dt),
      u_(u0), g_(sys.coupling_size()), du_(sys.fast_size()), time_(ta) {
    compute_rates();
  }

  size_t ode_size() const { return u_.size(); }
  double ode_time() const { return time_; }

  template <class It> It set_ode_state(It it, double t) {
    time_ = t;
    for (auto& ui : u_) ui = *it++;
    compute_rates();
    return it;
  }
  template <class It> It ode_state(It it) const {
    for (const auto& ui : u_) *it++ = ui;
    return it;
  }
  template <class It> It ode_rates(It it) const {
    for (const auto& d : du_) *it++ = d;
    return it;
  }

private:
  void compute_rates() {
    const double theta = (dt_ > 0.0) ? (time_ - ta_) / dt_ : 0.0;
    for (size_t d = 0; d < g_.size(); ++d) {
      value_type s = a_poly_[0][d];
      double p = theta;
      for (size_t m = 1; m < a_poly_.size(); ++m) { s += a_poly_[m][d] * p; p *= theta; }
      g_[d] = s;
    }
    sys_.fast_rates(u_, g_, du_);
  }

  const System& sys_;
  std::vector<vec> a_poly_;   // held by value: the leg's coupling is tiny and its
                              // lifetime must not depend on the caller's scratch
  double ta_, dt_;
  vec u_, g_, du_;
  double time_;
};

// The fast sub-cycle schedule: for each leg, the accepted step sizes. A double
// adaptive pass fills it (record); a later pass at any scalar replays it with
// fixed steps, so both passes take identical steps and a tape of the replay is
// the exact discrete adjoint of the scheme as run.
struct MRISchedule {
  std::vector<std::vector<double>> legs;
  size_t cursor = 0;
};

// Advance the fast block over one leg. replay=false records an adaptive schedule
// (double only); replay=true consumes the next recorded leg with fixed steps.
template <class System>
void advance_fast_leg(const System& sys, std::vector<typename System::value_type>& u,
                      const std::vector<std::vector<typename System::value_type>>& a_poly,
                      double ta, double dt, const OdeControl& control,
                      MRISchedule& sched, bool replay) {
  using S = typename System::value_type;
  FastLeg<System> leg(sys, u, a_poly, ta, dt);
  SolverInternal<FastLeg<System>> inner(leg, control);
  if (replay) {
    const std::vector<double>& hs = sched.legs.at(sched.cursor++);
    std::vector<double> times;
    times.reserve(hs.size() + 1);
    double t = ta;
    times.push_back(t);
    for (double h : hs) times.push_back(t += h);
    inner.advance_fixed(leg, times);
  } else if constexpr (std::is_same<S, double>::value) {
    inner.advance_adaptive(leg, ta + dt);
    const std::vector<double> ts = inner.get_times();
    std::vector<double> hs;
    hs.reserve(ts.size() - 1);
    for (size_t i = 1; i < ts.size(); ++i) hs.push_back(ts[i] - ts[i - 1]);
    sched.legs.push_back(std::move(hs));
  } else {
    util::stop("MRI record pass must run at double");
  }
  u = inner.get_state();
}

// One MRI macro step over [t, t+H]: advance slow block x and fast block u.
template <class System>
void mri_macro_step(const System& sys, const MRICoupling& M, const OdeControl& control,
                    double t, double H,
                    std::vector<typename System::value_type>& x,
                    std::vector<typename System::value_type>& u,
                    MRISchedule& sched, bool replay) {
  using S = typename System::value_type;
  const int n = M.nodes;
  const size_t nx = sys.slow_size(), cdim = sys.coupling_size();
  auto agg = [&](const std::vector<S>& v) {
    std::vector<S> g(cdim); sys.aggregate(v, g); return g;
  };

  std::vector<std::vector<S>> F(n), Fbar(n);   // slow tendency + its aggregate at each node
  F[0].resize(nx); sys.slow_rates(x, u, F[0]); Fbar[0] = agg(F[0]);

  for (int i = 1; i < n; ++i) {
    const double dc = M.c[i] - M.c[i - 1];
    if (dc > 0.0) {
      std::vector<std::vector<S>> a_poly(M.nmat + 1, std::vector<S>(cdim, S(0.0)));
      a_poly[0] = agg(x);
      for (int k = 0; k < M.nmat; ++k)
        for (int j = 0; j < i; ++j)
          for (size_t d = 0; d < cdim; ++d)
            a_poly[k + 1][d] += H / (k + 1) * M.G[k][i][j] * Fbar[j][d];

      advance_fast_leg(sys, u, a_poly, t + M.c[i - 1] * H, dc * H, control, sched, replay);
    }
    for (size_t a = 0; a < nx; ++a) {
      S inc(0.0);
      for (int j = 0; j < i; ++j) inc += M.abar(i, j) * F[j][a];
      x[a] += H * inc;
    }
    if (i < n - 1) { F[i].resize(nx); sys.slow_rates(x, u, F[i]); Fbar[i] = agg(F[i]); }
  }
}

// Advance a multirate System over a macro grid (which must land on the forcing
// kinks). Returns the full state [u; x] at each grid point; leaves the System at
// the final state. On a double record pass the schedule is filled; on a replay
// pass at any scalar it is consumed.
template <class System>
std::vector<std::vector<typename System::value_type>>
mri_advance(System& sys, const MRICoupling& M, const OdeControl& control,
            const std::vector<double>& macro_times, MRISchedule& sched,
            bool replay = false) {
  using S = typename System::value_type;
  const size_t nf = sys.fast_size(), ns = sys.slow_size();
  std::vector<S> full(sys.ode_size());
  sys.ode_state(full.begin());
  std::vector<S> u(full.begin(), full.begin() + nf), x(full.begin() + nf, full.end());

  if (replay) sched.cursor = 0;   // a full pass always consumes legs from the start
  std::vector<std::vector<S>> hist;
  hist.push_back(full);
  for (size_t m = 0; m + 1 < macro_times.size(); ++m) {
    mri_macro_step(sys, M, control, macro_times[m], macro_times[m + 1] - macro_times[m],
                   x, u, sched, replay);
    std::vector<S> s;
    s.reserve(nf + ns);
    s.insert(s.end(), u.begin(), u.end());
    s.insert(s.end(), x.begin(), x.end());
    hist.push_back(s);
  }
  std::vector<S> s;
  s.insert(s.end(), u.begin(), u.end());
  s.insert(s.end(), x.begin(), x.end());
  internal::set_ode_state(sys, s, macro_times.back());
  return hist;
}

// Fast-substep count in a schedule (cost accounting for the forward/record pass).
inline long mri_fast_steps(const MRISchedule& sched) {
  long n = 0;
  for (const auto& leg : sched.legs) n += (long)leg.size();
  return n;
}

}
}

#endif
