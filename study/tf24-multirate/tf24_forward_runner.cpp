// Forward multi-rate demonstration on a TF24-STRUCTURE surrogate (double, no AD).
// Adds the coupling structure the smooth surrogate lacked, to de-risk the real-patch
// bring-up:
//   * divergent matric-potential read  psi_soil(theta) = -a_psi (theta/theta_sat)^-n  (capped)
//   * a NESTED collar-potential root (the leaf continuity solve): find P s.t.
//     sum_i k_i * max(0, psi_i - P) = transpiration demand D  -> per-layer uptake E_i,
//     with layer on/off branches (kinks) as layers cross the collar potential
//   * a HARD u_min positivity floor on soil moisture (the event/clamp)
//   * two-way coupling: canopy sets demand D; soil sets collar potential P -> canopy water status
// Compares a global adaptive RK45 run to the mode-flagged multi-rate mechanic (freeze the
// slow canopy block, sub-cycle the soil, advance the canopy once per macro step).
// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <chrono>
using namespace Rcpp;

struct FP {
  double theta_sat=0.428, K_sat=163.0411, n_psi=6.57, theta_res=1e-2, dz=300.0;
  double a_infil=1.0, b_infil=8.0, a_psi=1.78e-3 /*MPa*/, psi_cap=100.0 /*MPa*/;
  double k_root=6.0 /*mm/day/MPa*/, P_floor=-100.0 /*MPa*/, D0=5.0 /*mm/day*/, P50=2.0 /*MPa*/;
};
static const double ROOT[5]={0.35,0.28,0.20,0.12,0.05};

static double Kf(double th,const FP&p){ double r=std::max(th,0.0)/p.theta_sat; return p.K_sat*std::pow(r,2*p.n_psi+3); }
// divergent matric potential (negative MPa), magnitude capped
static double psi_soil(double th,const FP&p){ double t=std::max(th,p.theta_res)/p.theta_sat;
  double v=p.a_psi*std::pow(t,-p.n_psi); if(v>p.psi_cap)v=p.psi_cap; return -v; }
// canopy water status from collar potential (bounded vulnerability curve, 1 wet .. 0 stressed)
static double Wstat(double P,const FP&p){ double x=P/p.P50; return 1.0/(1.0+x*x); }  // P<=0

// Nested collar-potential root: sum_i k_i max(0, psi_i - P) = D. Supply S(P) is
// continuous, piecewise-linear, decreasing in P. Bisection; supply-limited at P_floor.
// Returns P and fills E[5] (per-layer uptake, mm/day).
static double collar_solve(const double th[5], double D, const FP&p, double E[5]){
  double psi[5]; for(int i=0;i<5;++i) psi[i]=psi_soil(th[i],p);
  auto S=[&](double P){ double s=0; for(int i=0;i<5;++i){ double d=psi[i]-P; if(d>0) s+=p.k_root*ROOT[i]*d; } return s; };
  double Phi=0.0;                 // S(0) is the max (all layers, if psi_i<0 none supply at P=0)
  // upper bracket: P just below max psi (there S>0); use max(psi) as the zero-supply point
  double Pmax=psi[0]; for(int i=1;i<5;++i) Pmax=std::max(Pmax,psi[i]);
  double Plo=p.P_floor, Phi_b=Pmax;      // S(Plo) large, S(Phi_b)=0
  double P;
  if (S(Plo) <= D) { P=Plo; }            // supply-limited: even at the floor, supply < demand
  else {
    for(int it=0; it<80; ++it){ double Pm=0.5*(Plo+Phi_b); if(S(Pm)>D) Plo=Pm; else Phi_b=Pm; }
    P=0.5*(Plo+Phi_b);
  }
  for(int i=0;i<5;++i){ double d=psi[i]-P; E[i]=(d>0)?p.k_root*ROOT[i]*d:0.0; }
  (void)Phi; return P;
}

// soil rates (5) + collar potential P, given demand D and rainfall flux
static double soil_rate(const double th[5], double D, double rain, const FP&p, double dth[5]){
  double E[5]; double P=collar_solve(th,D,p,E);
  double r0=std::max(th[0],0.0)/p.theta_sat;
  double runoff=1.0-p.a_infil*std::pow(r0,p.b_infil); if(runoff<0)runoff=0;
  double infil=rain*runoff, out[5]; for(int i=0;i<5;++i)out[i]=Kf(th[i],p);
  for(int i=0;i<5;++i){ double win=(i==0)?infil:out[i-1];
    double r=(win-out[i]-E[i])/p.dz;
    if (th[i]<=p.theta_res && r<0) r=0.0;   // HARD u_min positivity floor (the event)
    dth[i]=r; }
  return P;
}
static double rain_at(const std::vector<double>&rain,double t){ long d=(long)std::floor(t); if(d<0)d=0; if(d>=(long)rain.size())d=rain.size()-1; return rain[d]; }

// Cash-Karp coefficients
namespace ck{ const double c2=.2,c3=.3,c4=.6,c5=1,c6=.875,
 a21=.2,a31=3./40,a32=9./40,a41=.3,a42=-.9,a43=1.2,a51=-11./54,a52=2.5,a53=-70./27,a54=35./27,
 a61=1631./55296,a62=175./512,a63=575./13824,a64=44275./110592,a65=253./4096,
 b1=37./378,b3=250./621,b4=125./594,b6=512./1771,
 e1=b1-2825./27648,e3=b3-18575./48384,e4=b4-13525./55296,e5=-277./14336,e6=b6-.25; }

// full-system RHS: [theta(5), canopy(M)]
static void full_rhs(const std::vector<double>&y,double t,int M,const std::vector<double>&alpha,
                     const std::vector<double>&rain,const FP&p,std::vector<double>&dy){
  double th[5]; for(int i=0;i<5;++i)th[i]=y[i];
  double cbar=0; for(int k=0;k<M;++k)cbar+=y[5+k]; cbar=(M>0)?cbar/M:0.0;
  double D=p.D0*(0.3+0.7*cbar);
  double dth[5]; double P=soil_rate(th,D,rain_at(rain,t),p,dth);
  for(int i=0;i<5;++i)dy[i]=dth[i];
  double W=Wstat(P,p); for(int k=0;k<M;++k)dy[5+k]=alpha[k]*(W-y[5+k]);
}

// adaptive RKCK on the full system over [t0,t1] at fixed output grid; returns steps + traj
// [[Rcpp::export]]
List global_run(std::vector<double> rain, std::vector<double> grid, int M,
                double tol_rel, double tol_abs, double D0){
  FP p; p.D0=D0; int N=5+M; std::vector<double> alpha(M);
  for(int k=0;k<M;++k) alpha[k]=0.02+0.08*((M>1)?double(k)/(M-1):0.0);
  std::vector<double> y(N,0.0); for(int i=0;i<5;++i)y[i]=0.30*p.theta_sat; for(int k=0;k<M;++k)y[5+k]=0.5;
  size_t ng=grid.size(); NumericMatrix theta(ng,5); NumericVector cbar(ng);
  auto rec=[&](size_t r){ for(int i=0;i<5;++i)theta(r,i)=y[i]; double cb=0; for(int k=0;k<M;++k)cb+=y[5+k]; cbar[r]=(M>0)?cb/M:0.0; };
  rec(0);
  std::vector<double> k1(N),k2(N),k3(N),k4(N),k5(N),k6(N),tmp(N),y5(N);
  long steps=0; double hmax=1.0;
  auto t0=std::chrono::high_resolution_clock::now();
  for(size_t gi=0; gi+1<ng; ++gi){
    double t=grid[gi], tend=grid[gi+1], h=std::min(hmax,tend-t); int guard=0;
    while(t<tend-1e-12 && guard++<500000){ if(t+h>tend)h=tend-t;
      full_rhs(y,t,M,alpha,rain,p,k1);
      for(int i=0;i<N;++i)tmp[i]=y[i]+h*ck::a21*k1[i]; full_rhs(tmp,t+ck::c2*h,M,alpha,rain,p,k2);
      for(int i=0;i<N;++i)tmp[i]=y[i]+h*(ck::a31*k1[i]+ck::a32*k2[i]); full_rhs(tmp,t+ck::c3*h,M,alpha,rain,p,k3);
      for(int i=0;i<N;++i)tmp[i]=y[i]+h*(ck::a41*k1[i]+ck::a42*k2[i]+ck::a43*k3[i]); full_rhs(tmp,t+ck::c4*h,M,alpha,rain,p,k4);
      for(int i=0;i<N;++i)tmp[i]=y[i]+h*(ck::a51*k1[i]+ck::a52*k2[i]+ck::a53*k3[i]+ck::a54*k4[i]); full_rhs(tmp,t+ck::c5*h,M,alpha,rain,p,k5);
      for(int i=0;i<N;++i)tmp[i]=y[i]+h*(ck::a61*k1[i]+ck::a62*k2[i]+ck::a63*k3[i]+ck::a64*k4[i]+ck::a65*k5[i]); full_rhs(tmp,t+ck::c6*h,M,alpha,rain,p,k6);
      double err=0; for(int i=0;i<N;++i){ y5[i]=y[i]+h*(ck::b1*k1[i]+ck::b3*k3[i]+ck::b4*k4[i]+ck::b6*k6[i]);
        double e=h*(ck::e1*k1[i]+ck::e3*k3[i]+ck::e4*k4[i]+ck::e5*k5[i]+ck::e6*k6[i]);
        double sc=tol_abs+tol_rel*std::max(std::fabs(y[i]),std::fabs(y5[i])); err+=(e/sc)*(e/sc); }
      err=std::sqrt(err/N); if(!std::isfinite(err))err=1e3;
      if(err<=1){ y=y5; t+=h; ++steps; double f=std::min(5.,std::max(.2,.9*std::pow(err+1e-30,-.2))); h=std::min(hmax,h*f); }
      else { h*=std::min(1.,std::max(.1,.9*std::pow(err,-.2))); if(h<1e-9){++steps; t+=h;} } }
    rec(gi+1);
  }
  auto t1=std::chrono::high_resolution_clock::now();
  return List::create(_["theta"]=theta,_["cbar"]=cbar,_["steps"]=(double)steps,
    _["wall_ms"]=std::chrono::duration<double,std::milli>(t1-t0).count());
}

// mode-flagged multi-rate: per macro step, FREEZE canopy (demand D fixed) and sub-cycle the
// 5 soil states adaptively (each micro-step does the collar solve, reads divergent psi, honours
// the u_min floor), accumulating the time-averaged canopy water status; then advance the canopy
// once per macro step. This is the real-patch mechanic (cohort rates frozen during the sub-cycle).
// [[Rcpp::export]]
List mrate_run(std::vector<double> rain, std::vector<double> macro, int M,
               double tol_rel, double tol_abs, double D0){
  FP p; p.D0=D0; std::vector<double> alpha(M);
  for(int k=0;k<M;++k) alpha[k]=0.02+0.08*((M>1)?double(k)/(M-1):0.0);
  std::vector<double> th(5); for(int i=0;i<5;++i)th[i]=0.30*p.theta_sat;
  std::vector<double> c(M,0.5);
  size_t ng=macro.size(); NumericMatrix theta(ng,5); NumericVector cbar(ng);
  auto rec=[&](size_t r){ for(int i=0;i<5;++i)theta(r,i)=th[i]; double cb=0; for(int k=0;k<M;++k)cb+=c[k]; cbar[r]=(M>0)?cb/M:0.0; };
  rec(0);
  long soil_steps=0; double hmax=0.5;
  auto t0=std::chrono::high_resolution_clock::now();
  for(size_t m=0;m+1<ng;++m){
    double ta=macro[m], tb=macro[m+1], H=tb-ta;
    double cb=0; for(int k=0;k<M;++k)cb+=c[k]; cb=(M>0)?cb/M:0.0;
    double D=p.D0*(0.3+0.7*cb);             // frozen demand over the macro step
    // sub-cycle soil (5 states) adaptive RKCK; accumulate integral of Wstat(P)
    double t=ta, h=std::min(hmax,H); int guard=0; double Wint=0.0;
    std::vector<double> y=th, k1(5),k2(5),k3(5),k4(5),k5(5),k6(5),tmp(5),y5(5);
    auto rate=[&](const std::vector<double>&yy,double tt,std::vector<double>&d){
      double a[5]; for(int i=0;i<5;++i)a[i]=yy[i]; double dd[5];
      double P=soil_rate(a,D,rain_at(rain,tt),p,dd); for(int i=0;i<5;++i)d[i]=dd[i]; return P; };
    while(t<tb-1e-12 && guard++<500000){ if(t+h>tb)h=tb-t;
      double P1=rate(y,t,k1);
      for(int i=0;i<5;++i)tmp[i]=y[i]+h*ck::a21*k1[i]; rate(tmp,t+ck::c2*h,k2);
      for(int i=0;i<5;++i)tmp[i]=y[i]+h*(ck::a31*k1[i]+ck::a32*k2[i]); rate(tmp,t+ck::c3*h,k3);
      for(int i=0;i<5;++i)tmp[i]=y[i]+h*(ck::a41*k1[i]+ck::a42*k2[i]+ck::a43*k3[i]); rate(tmp,t+ck::c4*h,k4);
      for(int i=0;i<5;++i)tmp[i]=y[i]+h*(ck::a51*k1[i]+ck::a52*k2[i]+ck::a53*k3[i]+ck::a54*k4[i]); rate(tmp,t+ck::c5*h,k5);
      for(int i=0;i<5;++i)tmp[i]=y[i]+h*(ck::a61*k1[i]+ck::a62*k2[i]+ck::a63*k3[i]+ck::a64*k4[i]+ck::a65*k5[i]); rate(tmp,t+ck::c6*h,k6);
      double err=0; for(int i=0;i<5;++i){ y5[i]=y[i]+h*(ck::b1*k1[i]+ck::b3*k3[i]+ck::b4*k4[i]+ck::b6*k6[i]);
        double e=h*(ck::e1*k1[i]+ck::e3*k3[i]+ck::e4*k4[i]+ck::e5*k5[i]+ck::e6*k6[i]);
        double sc=tol_abs+tol_rel*std::max(std::fabs(y[i]),std::fabs(y5[i])); err+=(e/sc)*(e/sc); }
      err=std::sqrt(err/5); if(!std::isfinite(err))err=1e3;
      if(err<=1){ // trapezoidal accumulation of Wstat over the accepted step
        double E[5]; double Pend=collar_solve(&y5[0],D,p,E);
        Wint += 0.5*h*(Wstat(P1,p)+Wstat(Pend,p));
        y=y5; t+=h; ++soil_steps; double f=std::min(5.,std::max(.2,.9*std::pow(err+1e-30,-.2))); h=std::min(hmax,h*f); }
      else { h*=std::min(1.,std::max(.1,.9*std::pow(err,-.2))); if(h<1e-9){ double E[5]; collar_solve(&y[0],D,p,E); ++soil_steps; t+=h; } } }
    for(int i=0;i<5;++i)th[i]=y[i];
    double Wbar=(H>0)?Wint/H:Wstat( 0.0,p);
    for(int k=0;k<M;++k) c[k]=Wbar+(c[k]-Wbar)*std::exp(-alpha[k]*H);  // exact canopy advance
    rec(m+1);
  }
  auto t1=std::chrono::high_resolution_clock::now();
  return List::create(_["theta"]=theta,_["cbar"]=cbar,_["soil_steps"]=(double)soil_steps,
    _["wall_ms"]=std::chrono::duration<double,std::milli>(t1-t0).count());
}

// probe the collar solve across a moisture range (diagnostic: divergent psi + layer on/off)
// [[Rcpp::export]]
List collar_probe(std::vector<double> thetas, double D, double D0){
  FP p; p.D0=D0; int n=thetas.size(); NumericVector P(n),Etot(n),psi1(n),nact(n);
  for(int j=0;j<n;++j){ double th[5]; for(int i=0;i<5;++i)th[i]=thetas[j];
    double E[5]; P[j]=collar_solve(th,D,p,E); double s=0; int na=0;
    for(int i=0;i<5;++i){s+=E[i]; if(E[i]>0)na++;} Etot[j]=s; nact[j]=na; psi1[j]=psi_soil(thetas[j],p); }
  return List::create(_["P"]=P,_["Etot"]=Etot,_["psi"]=psi1,_["n_active"]=nact);
}
