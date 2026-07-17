// MRI core self-tests: (1) row sums, (2) collapse-to-base-ERK (f^F=0),
// (3) collapse-to-inner (f^S=0), (4) empirical coupling order on a linear two-rate
// system with a tight reference. These are the "sharp tests" the brief calls out.
#include "mri_core.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>

// base explicit RK step (for the collapse-1 oracle)
static void erk_step(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
                     const std::vector<double>& c, std::function<double(double,double)> f,
                     double& x, double t, double H) {
  int s = (int)b.size(); std::vector<double> k(s, 0.0);
  for (int i = 0; i < s; ++i) {
    double xi = x; for (int j = 0; j < i; ++j) xi += H * A[i][j] * k[j];
    k[i] = f(t + c[i] * H, xi);
  }
  double inc = 0.0; for (int i = 0; i < s; ++i) inc += b[i] * k[i];
  x += H * inc;
}

// tight fixed-step RK4 of the full 2D linear system (reference / inner)
struct Lin { double a, w, b, d; };
static void rk4_full(const Lin& P, double& x, double& u, double t0, double t1, long n) {
  double h = (t1 - t0) / n;
  auto fx = [&](double X, double U){ return -P.a * X + P.b * U; };
  auto fu = [&](double X, double U){ return -P.w * U + P.d * X; };
  for (long i = 0; i < n; ++i) {
    double kx1=fx(x,u), ku1=fu(x,u);
    double kx2=fx(x+0.5*h*kx1,u+0.5*h*ku1), ku2=fu(x+0.5*h*kx1,u+0.5*h*ku1);
    double kx3=fx(x+0.5*h*kx2,u+0.5*h*ku2), ku3=fu(x+0.5*h*kx2,u+0.5*h*ku2);
    double kx4=fx(x+h*kx3,u+h*ku3), ku4=fu(x+h*kx3,u+h*ku3);
    x += h/6*(kx1+2*kx2+2*kx3+kx4); u += h/6*(ku1+2*ku2+2*ku3+ku4);
  }
}

int main() {
  printf("=== MRI core self-tests ===\n\n");
  std::vector<MRICoupling> tabs = {table_fwd_euler(), table_midpoint(), table_heun(), table_kutta3()};

  // (1) row sums
  printf("(1) consistency (row sums = Δc, Σb=1):\n");
  for (auto& M : tabs) printf("  %-16s q=%d  %s\n", M.name, M.q, check_rowsums(M) ? "OK" : "FAIL");

  Lin P{0.5, 20.0, 0.3, 0.4};
  double T = 2.0, x0 = 1.0, u0 = 0.5;

  // reference
  double xr = x0, ur = u0; rk4_full(P, xr, ur, 0, T, 1<<21);

  // the fast inner used by the MRI driver: integrate u'=-ω u + d·x̄(τ) tightly.
  auto make_inner = [&](int n_per_leg){
    return [=,&P](std::vector<double>& u, const std::vector<double>& poly, double ta, double tb){
      double dt = tb - ta, h = dt / n_per_leg, U = u[0];
      auto xbar = [&](double th){ double tau = th/dt, s=0, p=1; for(double c:poly){ s+=c*p; p*=tau; } return s; };
      auto fu = [&](double th, double U_){ return -P.w*U_ + P.d*xbar(th); };
      for (int i=0;i<n_per_leg;++i){ double th=i*h;
        double k1=fu(th,U), k2=fu(th+0.5*h,U+0.5*h*k1), k3=fu(th+0.5*h,U+0.5*h*k2), k4=fu(th+h,U+h*k3);
        U += h/6*(k1+2*k2+2*k3+k4); }
      u[0]=U;
    };
  };
  auto slow = [&](const std::vector<double>& x, const std::vector<double>& u){
    return std::vector<double>{ -P.a*x[0] + P.b*u[0] }; };
  auto agg  = [](const std::vector<double>& v){ return v[0]; };

  // (2) collapse to base ERK when f^F = 0 (fast frozen): MRI x-traj == base ERK.
  printf("\n(2) collapse to induced base ERK (f^F=0), max|Δx| over 8 steps:\n");
  std::vector<std::pair<MRICoupling,std::pair<std::vector<std::vector<double>>,std::pair<std::vector<double>,std::vector<double>>>>> bases;
  auto run_collapse1 = [&](const MRICoupling& M, const std::vector<std::vector<double>>& A,
                           const std::vector<double>& b, const std::vector<double>& c){
    double H=0.1; int nst=8;
    // slow-only field: x' = sin(x) (nonlinear, exercises stages); fast frozen
    auto fslow1 = [](double t,double X){ return std::sin(X)+0.3; };
    auto slow1 = [&](const std::vector<double>& x, const std::vector<double>&){ return std::vector<double>{ std::sin(x[0])+0.3 }; };
    auto inner0 = [](std::vector<double>&, const std::vector<double>&, double, double){}; // u fixed
    std::vector<double> xM{0.2}, uM{0.0}; double xE=0.2;
    double maxd=0;
    for (int n=0;n<nst;++n){ mri_macro_step(M, n*H, H, xM, uM, slow1, agg, inner0);
      erk_step(A,b,c, fslow1, xE, n*H, H); maxd=std::max(maxd, std::fabs(xM[0]-xE)); }
    printf("  %-16s %.2e\n", M.name, maxd);
  };
  run_collapse1(table_fwd_euler(), {{0.0}}, {1.0}, {0.0});
  run_collapse1(table_midpoint(), {{0,0},{0.5,0}}, {0,1}, {0,0.5});
  run_collapse1(table_heun(), {{0,0},{1,0}}, {0.5,0.5}, {0,1});
  run_collapse1(table_kutta3(), {{0,0,0},{0.5,0,0},{-1,2,0}}, {1.0/6,2.0/3,1.0/6}, {0,0.5,1});

  // (3) collapse to pure inner when f^S = 0: MRI u == single tight inner over [0,H].
  printf("\n(3) collapse to pure inner (f^S=0), max|Δu| over 8 steps:\n");
  for (auto& M : tabs) {
    double H=0.1; int nst=8; auto slow0=[&](const std::vector<double>&,const std::vector<double>&){ return std::vector<double>{0.0}; };
    auto inner = make_inner(500);
    std::vector<double> xM{1.0}, uM{0.5}; double maxd=0;
    for (int n=0;n<nst;++n){ double uref=uM[0];
      // pure inner over [nH,(n+1)H] with x̄ = const = xM (since slow=0, x frozen)
      std::vector<double> up{uref}; std::vector<double> poly{xM[0]}; inner(up, poly, n*H,(n+1)*H);
      mri_macro_step(M, n*H, H, xM, uM, slow0, agg, inner);
      maxd=std::max(maxd, std::fabs(uM[0]-up[0])); }
    printf("  %-16s %.2e\n", M.name, maxd);
  }

  // (4) empirical coupling order (linear two-rate, tight inner)
  printf("\n(4) empirical coupling order (linear two-rate, ω=%.0f, tight inner):\n", P.w);
  printf("  %-16s %10s %10s %10s %10s %10s\n","table","H=T/2","T/4","T/8","T/16","order");
  for (auto& M : tabs) {
    std::vector<double> errs;
    for (int nmac : {2,4,8,16,32}) {
      double H=T/nmac; std::vector<double> x{x0}, u{u0}; auto inner=make_inner(600);
      for (int n=0;n<nmac;++n) mri_macro_step(M, n*H, H, x, u, slow, agg, inner);
      errs.push_back(std::max(std::fabs(x[0]-xr), std::fabs(u[0]-ur)));
    }
    // order estimate from last two refinements
    double ord = std::log2(errs[3]/errs[4]);
    printf("  %-16s %10.2e %10.2e %10.2e %10.2e %10.2f\n", M.name, errs[0],errs[1],errs[2],errs[3], ord);
  }
  printf("\n(reference via RK4, 2^21 steps)\n");
  return 0;
}
