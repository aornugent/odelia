// Phase-B prototype: reverse-mode AD through the multirate MRI scheme on the
// two-way coupled soil+canopy surrogate (TF24/TF24f structure: 5 soil states + M
// smooth canopy states + a trait vector). Two-level frozen schedule: a double
// adaptive pass records the per-leg micro step sizes; an active pass replays them
// (fixed steps, no adaptive branching) so the tape of the scheme-as-run is the
// discrete adjoint. Templated on the scalar S so it runs at double or AReal<double>.
#ifndef MRI_AD_HPP
#define MRI_AD_HPP
#include <vector>
#include <cmath>
#include <XAD/XAD.hpp>
#include "mri_core.hpp"

// pow for double and AD scalars (base clamped > 0 by the caller).
template <class S> static inline S powS(const S& a, const S& b) {
  using std::pow; using xad::pow; return pow(a, b);
}

// Differentiable traits (the "trait combination" axis). Fixed soil geometry is
// constant; these five enter the coupled dynamics nonlinearly.
//   0 Ksat_mult  (drainage scale)      1 n_psi       (retention exponent -> p_drain)
//   2 t_pot      (transpiration demand) 3 theta_wilt  (uptake stress floor)
//   4 alpha_scale(canopy response rate)
template <class S> struct Traits {
  S Ksat_mult, n_psi, t_pot, theta_wilt, alpha_scale;
};
struct FixedPars {
  double theta_sat=0.428, K_sat=163.0411, theta_res=1e-2, depth_mm=1500.0,
         a_infil=1.0, b_infil=8.0, dz=300.0, stiff=1.0;
  double sdelta=0.0;   // 0 = hard clamps (near-bound non-smoothness); >0 = smooth floors
};
// smooth lower-bound max(a,b): C-infinity for delta>0, hard max at delta=0.
template <class S> static inline S sfloor(const S& a, double b, double delta) {
  using std::sqrt; using xad::sqrt;
  if (delta <= 0.0) { return (a < b) ? S(b) : a; }
  S d = a - b; S rad = d*d + delta*delta; return S(b) + S(0.5)*(d + sqrt(rad));
}

// soil rate for the 5 layers, reading the canopy aggregate xbar and rain flux.
template <class S>
static void soil_rhs(const std::vector<S>& th, const S& xbar, const Traits<S>& tr,
                     double rain_flux, const FixedPars& fp, std::vector<S>& dth) {
  static const double root[5] = {0.35,0.28,0.20,0.12,0.05};
  const int nL = 5;
  const double rfloor = fp.theta_res/fp.theta_sat;
  S p_drain = S(2.0)*tr.n_psi + S(3.0);
  S Tpot = tr.t_pot * (S(0.5) + xbar);
  S r0raw = th[0]/fp.theta_sat; S r0 = sfloor(r0raw, rfloor, fp.sdelta);
  S runoff_raw = S(1.0) - fp.a_infil*powS(r0, S(fp.b_infil));
  S runoff = sfloor(runoff_raw, 0.0, fp.sdelta);
  S infil = rain_flux * runoff;
  std::vector<S> out(nL);
  for (int i=0;i<nL;++i){ S rraw = th[i]/fp.theta_sat; S r = sfloor(rraw, rfloor, fp.sdelta);
    out[i] = fp.stiff*tr.Ksat_mult*fp.K_sat*powS(r, p_drain); }
  for (int i=0;i<nL;++i){
    S win = (i==0)? infil : out[i-1];
    // smooth uptake stress: 0 at residual, 1 at/above wilting
    S s = (th[i]-fp.theta_res)/(tr.theta_wilt-fp.theta_res);
    S stress = (s<=0)? S(0.0) : (s>=1)? S(1.0) : s*s*(S(3.0)-S(2.0)*s);
    dth[i] = (win - out[i] - Tpot*root[i]*stress)/fp.dz;
  }
}
// soil aggregate the canopy reads: Sbar = mean(theta)/theta_sat
template <class S> static S soil_agg(const std::vector<S>& th, const FixedPars& fp) {
  S s=S(0.0); for (auto& t:th) s+=t; return s/double(th.size())/fp.theta_sat;
}

// ---- RKCK (Cash-Karp) coefficients ----------------------------------------
namespace rkck {
  static constexpr double c2=0.2,c3=0.3,c4=0.6,c5=1.0,c6=0.875;
  static constexpr double a21=0.2, a31=3.0/40,a32=9.0/40, a41=0.3,a42=-0.9,a43=1.2,
    a51=-11.0/54,a52=2.5,a53=-70.0/27,a54=35.0/27,
    a61=1631.0/55296,a62=175.0/512,a63=575.0/13824,a64=44275.0/110592,a65=253.0/4096;
  static constexpr double b1=37.0/378,b3=250.0/621,b4=125.0/594,b6=512.0/1771;
  static constexpr double e1=b1-2825.0/27648,e3=b3-18575.0/48384,e4=b4-13525.0/55296,
    e5=-277.0/14336,e6=b6-0.25;
}

// one RKCK stage evaluation of the soil block at (th, t) reading xbar(tau).
template <class S, class XbarFn>
static void soil_deriv(const std::vector<S>& th, double t, XbarFn xbar, const Traits<S>& tr,
                       const std::vector<double>& rain, const FixedPars& fp, std::vector<S>& dth) {
  long d=(long)std::floor(t); if(d<0)d=0; if(d>=(long)rain.size())d=rain.size()-1;
  soil_rhs(th, xbar(t), tr, rain[d], fp, dth);
}

// Fixed-step RKCK over recorded micro steps `hs` on [ta,tb]; scalar S (tapes).
template <class S>
static void inner_replay(std::vector<S>& y, const std::vector<double>& hs, double ta, double tb,
                         const std::vector<S>& poly, const Traits<S>& tr,
                         const std::vector<double>& rain, const FixedPars& fp) {
  using namespace rkck; int n=(int)y.size();
  double dt = tb-ta;
  auto xbar=[&](double t){ double tau=(dt>0)?(t-ta)/dt:0.0; S s=poly[0]; double p=tau; for(size_t m=1;m<poly.size();++m){ s+=poly[m]*p; p*=tau; } return s; };
  double t=ta;
  std::vector<S> k1(n),k2(n),k3(n),k4(n),k5(n),k6(n),tmp(n);
  for (double h : hs) {
    soil_deriv(y,t,xbar,tr,rain,fp,k1);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*a21*k1[i]; soil_deriv(tmp,t+c2*h,xbar,tr,rain,fp,k2);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a31*k1[i]+a32*k2[i]); soil_deriv(tmp,t+c3*h,xbar,tr,rain,fp,k3);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a41*k1[i]+a42*k2[i]+a43*k3[i]); soil_deriv(tmp,t+c4*h,xbar,tr,rain,fp,k4);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]); soil_deriv(tmp,t+c5*h,xbar,tr,rain,fp,k5);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]); soil_deriv(tmp,t+c6*h,xbar,tr,rain,fp,k6);
    for(int i=0;i<n;++i) y[i]+=h*(b1*k1[i]+b3*k3[i]+b4*k4[i]+b6*k6[i]);
    t+=h;
  }
}

// Adaptive RKCK on [ta,tb] (double), recording accepted step sizes into `hs`.
static void inner_adaptive(std::vector<double>& y, double ta, double tb,
                           const std::vector<double>& poly, const Traits<double>& tr,
                           const std::vector<double>& rain, const FixedPars& fp,
                           double tol_rel, double tol_abs, double hmax, std::vector<double>& hs) {
  using namespace rkck; int n=(int)y.size(); double dt=tb-ta;
  auto xbar=[&](double t){ double tau=(dt>0)?(t-ta)/dt:0.0,s=poly[0],p=tau; for(size_t m=1;m<poly.size();++m){ s+=poly[m]*p; p*=tau; } return s; };
  double t=ta, h=std::min(hmax, dt>0?dt:hmax);
  std::vector<double> k1(n),k2(n),k3(n),k4(n),k5(n),k6(n),tmp(n),y5(n);
  int guard=0;
  while (t < tb-1e-12 && guard++ < 200000) {
    if (t+h > tb) h = tb-t;
    soil_deriv(y,t,xbar,tr,rain,fp,k1);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*a21*k1[i]; soil_deriv(tmp,t+c2*h,xbar,tr,rain,fp,k2);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a31*k1[i]+a32*k2[i]); soil_deriv(tmp,t+c3*h,xbar,tr,rain,fp,k3);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a41*k1[i]+a42*k2[i]+a43*k3[i]); soil_deriv(tmp,t+c4*h,xbar,tr,rain,fp,k4);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]); soil_deriv(tmp,t+c5*h,xbar,tr,rain,fp,k5);
    for(int i=0;i<n;++i) tmp[i]=y[i]+h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]); soil_deriv(tmp,t+c6*h,xbar,tr,rain,fp,k6);
    double err=0;
    for(int i=0;i<n;++i){ y5[i]=y[i]+h*(b1*k1[i]+b3*k3[i]+b4*k4[i]+b6*k6[i]);
      double e=h*(e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]);
      double sc=tol_abs+tol_rel*std::max(std::fabs(y[i]),std::fabs(y5[i])); err+=(e/sc)*(e/sc); }
    err=std::sqrt(err/n); if(!std::isfinite(err)) err=1e3;
    if (err<=1.0) { for(int i=0;i<n;++i) y[i]=y5[i]; t+=h; hs.push_back(h);
      double fac=std::min(5.0,std::max(0.2,0.9*std::pow(err+1e-30,-0.2))); h=std::min(hmax,h*fac); }
    else { h*=std::min(1.0,std::max(0.1,0.9*std::pow(err,-0.2))); if(h<1e-10){ hs.push_back(tb-t>0?tb-t:1e-10); t=tb; } }
  }
}

// Recorded schedule: per macro step m, per leg (Δc>0), the vector of micro step sizes.
struct MRISchedule { std::vector<std::vector<std::vector<double>>> h; };

// ---- MRI replay in scalar S over a frozen schedule; returns the functional ----
// Functional F = mean over macro sample points of the canopy aggregate x̄ (a smooth
// "canopy status" scalar depending on all traits through the coupled dynamics).
template <class S>
static S mri_replay(const MRICoupling& MT, const Traits<S>& tr, int M,
                    const std::vector<double>& macro_times, const std::vector<double>& rain,
                    const FixedPars& fp, double alpha_lo, double alpha_hi,
                    const MRISchedule& sched) {
  const int nL=5; int s=MT.s;
  std::vector<double> alpha(M);
  for(int k=0;k<M;++k) alpha[k]=alpha_lo+(alpha_hi-alpha_lo)*((M>1)?double(k)/(M-1):0.0);
  std::vector<S> x(M, S(0.30)), u(nL, S(0.30*fp.theta_sat));
  auto aggx=[&](const std::vector<S>& v){ S t=S(0.0); for(auto&e:v)t+=e; return v.empty()?S(0.0):t/double(v.size()); };
  auto slow=[&](const std::vector<S>& canopy, const std::vector<S>& soil){
    S Sbar=soil_agg(soil,fp); std::vector<S> d(M);
    for(int k=0;k<M;++k) d[k]=tr.alpha_scale*alpha[k]*(Sbar-canopy[k]); return d; };
  S Fsum = aggx(x); size_t nsamp=1;
  size_t ng=macro_times.size();
  for(size_t m=0;m+1<ng;++m){
    double t0=macro_times[m], H=macro_times[m+1]-macro_times[m];
    std::vector<std::vector<S>> F(s+1); std::vector<S> Fbar(s+1, S(0.0));
    F[1]=slow(x,u); Fbar[1]=aggx(F[1]);
    std::vector<S> zx=x, zu=u; int legidx=0;
    for(int i=2;i<=s+1;++i){
      double dc=MT.c[i-1]-MT.c[i-2];
      if(dc<=0){ for(int a=0;a<M;++a){ S inc=S(0.0); for(int j=1;j<i;++j) inc+=MT.abar(i-1,j-1)*F[j][a]; zx[a]+=H*inc; } }
      else {
        std::vector<S> poly(MT.K+2, S(0.0)); poly[0]=aggx(zx);
        for(int k=0;k<=MT.K;++k){ S sj=S(0.0); for(int j=1;j<i;++j) sj+=MT.G[k][i-1][j-1]*Fbar[j]; poly[k+1]+=H*sj/(k+1); }
        double ta=t0+MT.c[i-2]*H, tb=t0+MT.c[i-1]*H;
        inner_replay(zu, sched.h[m][legidx++], ta, tb, poly, tr, rain, fp);
        for(int a=0;a<M;++a){ S inc=S(0.0); for(int j=1;j<i;++j) inc+=MT.abar(i-1,j-1)*F[j][a]; zx[a]+=H*inc; }
      }
      if(i<=s){ F[i]=slow(zx,zu); Fbar[i]=aggx(F[i]); }
    }
    x=zx; u=zu; Fsum += aggx(x); ++nsamp;
  }
  return Fsum/double(nsamp);
}

// ---- record pass (double, adaptive) → functional + schedule -------------------
static double mri_record(const MRICoupling& MT, const Traits<double>& tr, int M,
                         const std::vector<double>& macro_times, const std::vector<double>& rain,
                         const FixedPars& fp, double alpha_lo, double alpha_hi,
                         double tol_rel, double tol_abs, MRISchedule& sched, long& soil_steps) {
  const int nL=5; int s=MT.s;
  std::vector<double> alpha(M);
  for(int k=0;k<M;++k) alpha[k]=alpha_lo+(alpha_hi-alpha_lo)*((M>1)?double(k)/(M-1):0.0);
  std::vector<double> x(M,0.30), u(nL,0.30*fp.theta_sat);
  auto aggx=[&](const std::vector<double>& v){ double t=0; for(double e:v)t+=e; return v.empty()?0.0:t/v.size(); };
  auto slow=[&](const std::vector<double>& canopy, const std::vector<double>& soil){
    double Sbar=soil_agg(soil,fp); std::vector<double> d(M);
    for(int k=0;k<M;++k) d[k]=tr.alpha_scale*alpha[k]*(Sbar-canopy[k]); return d; };
  double Fsum=aggx(x); size_t nsamp=1; soil_steps=0;
  size_t ng=macro_times.size(); sched.h.assign(ng-1, {});
  double hmax=std::min(0.5, 0.5*(ng>1?macro_times[1]-macro_times[0]:1.0));
  for(size_t m=0;m+1<ng;++m){
    double t0=macro_times[m], H=macro_times[m+1]-macro_times[m];
    std::vector<std::vector<double>> F(s+1); std::vector<double> Fbar(s+1,0.0);
    F[1]=slow(x,u); Fbar[1]=aggx(F[1]);
    std::vector<double> zx=x, zu=u;
    for(int i=2;i<=s+1;++i){
      double dc=MT.c[i-1]-MT.c[i-2];
      if(dc<=0){ for(int a=0;a<M;++a){ double inc=0; for(int j=1;j<i;++j) inc+=MT.abar(i-1,j-1)*F[j][a]; zx[a]+=H*inc; } }
      else {
        std::vector<double> poly(MT.K+2,0.0); poly[0]=aggx(zx);
        for(int k=0;k<=MT.K;++k){ double sj=0; for(int j=1;j<i;++j) sj+=MT.G[k][i-1][j-1]*Fbar[j]; poly[k+1]+=H*sj/(k+1); }
        double ta=t0+MT.c[i-2]*H, tb=t0+MT.c[i-1]*H;
        std::vector<double> hs; inner_adaptive(zu, ta, tb, poly, tr, rain, fp, tol_rel, tol_abs, hmax, hs);
        soil_steps += (long)hs.size(); sched.h[m].push_back(hs);
        for(int a=0;a<M;++a){ double inc=0; for(int j=1;j<i;++j) inc+=MT.abar(i-1,j-1)*F[j][a]; zx[a]+=H*inc; }
      }
      if(i<=s){ F[i]=slow(zx,zu); Fbar[i]=aggx(F[i]); }
    }
    x=zx; u=zu; Fsum+=aggx(x); ++nsamp;
  }
  return Fsum/double(nsamp);
}
#endif
