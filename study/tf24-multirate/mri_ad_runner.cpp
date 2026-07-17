// Reverse-mode gradients through the multirate MRI scheme (Phase-B prototype).
// Records the two-level frozen schedule in double, replays under XAD reverse mode to
// tape the scheme-as-run, and returns the gradient + a frozen-schedule FD check.
// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <functional>
#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>
#include "mri_ad.hpp"

static MRICoupling pick_table(const std::string& n){
  if(n=="fwd_euler")return table_fwd_euler(); if(n=="midpoint")return table_midpoint();
  if(n=="heun")return table_heun(); if(n=="kutta3")return table_kutta3();
  return table_midpoint();
}
template<class S> static Traits<S> make_traits(const std::vector<S>& v){
  return Traits<S>{v[0],v[1],v[2],v[3],v[4]};
}
static double now_ms(){ return 0.0; }  // unused; chrono below

// Reverse-mode gradient of F(traits) + frozen-schedule central-FD check.
// trait_vals: 5 values [Ksat_mult,n_psi,t_pot,theta_wilt,alpha_scale].
// [[Rcpp::export]]
Rcpp::List mri_grad(std::vector<double> rain, std::vector<double> trait_vals, int M,
                    std::vector<double> macro_times, std::string table,
                    double tol_rel, double tol_abs, double alpha_lo, double alpha_hi,
                    double eps_fd, bool do_fd, std::vector<int> active, double smooth_delta) {
  using clk=std::chrono::high_resolution_clock;
  FixedPars fp; fp.sdelta=smooth_delta; MRICoupling MT=pick_table(table);
  if (active.empty()) { for (int i=0;i<(int)trait_vals.size();++i) active.push_back(i); }
  int na=(int)active.size();

  // ---- pass 1: record the frozen schedule (double, adaptive) ----
  MRISchedule sched; long soil_steps=0;
  auto trd=make_traits(trait_vals);
  auto t0=clk::now();
  double F0=mri_record(MT, trd, M, macro_times, rain, fp, alpha_lo, alpha_hi, tol_rel, tol_abs, sched, soil_steps);
  auto t1=clk::now();
  double wall_record=std::chrono::duration<double,std::milli>(t1-t0).count();

  // ---- forward replay cost (double) for the reverse/forward ratio ----
  t0=clk::now();
  double Frep=mri_replay(MT, trd, M, macro_times, rain, fp, alpha_lo, alpha_hi, sched);
  t1=clk::now(); double wall_fwd=std::chrono::duration<double,std::milli>(t1-t0).count();

  // ---- reverse-mode gradient (all traits at once, one tape sweep) ----
  using ad=xad::adj<double>; using AD=ad::active_type;
  ad::tape_type tape(false); tape.activate();
  std::vector<AD> inputs; for (int i:active) inputs.push_back(AD(trait_vals[i]));
  std::function<std::vector<AD>(std::vector<AD>&)> fwd=[&](std::vector<AD>& x){
    std::vector<AD> full(trait_vals.begin(), trait_vals.end());
    for (int k=0;k<na;++k) full[active[k]]=x[k];
    Traits<AD> tr=make_traits(full);
    AD F=mri_replay(MT, tr, M, macro_times, rain, fp, alpha_lo, alpha_hi, sched);
    return std::vector<AD>{F};
  };
  t0=clk::now();
  auto jac=xad::computeJacobian(inputs, fwd, (size_t)1, &tape);
  t1=clk::now(); double wall_rev=std::chrono::duration<double,std::milli>(t1-t0).count();
  tape.deactivate();
  std::vector<double> grad_rev(jac[0].begin(), jac[0].end());

  // ---- frozen-schedule central FD (same recorded schedule) ----
  std::vector<double> grad_fd(na, NA_REAL); double wall_fd=0;
  if (do_fd) {
    t0=clk::now();
    for (int k=0;k<na;++k){ int i=active[k];
      double e=eps_fd*std::max(1.0,std::fabs(trait_vals[i]));
      std::vector<double> vp=trait_vals, vm=trait_vals; vp[i]+=e; vm[i]-=e;
      double Fp=mri_replay(MT, make_traits(vp), M, macro_times, rain, fp, alpha_lo, alpha_hi, sched);
      double Fm=mri_replay(MT, make_traits(vm), M, macro_times, rain, fp, alpha_lo, alpha_hi, sched);
      grad_fd[k]=(Fp-Fm)/(2*e);
    }
    t1=clk::now(); wall_fd=std::chrono::duration<double,std::milli>(t1-t0).count();
  }
  // combined abs+rel error: |rev-fd| / (atol + rtol|fd|). Near-zero gradient
  // components (e.g. drainage in a bone-dry year) have huge *relative* error but
  // negligible absolute error; the tape is exact to the FD noise floor either way.
  const double atol=1e-7, rtol=1e-4;
  double max_rel=0, max_abs=0, norm_err=0;
  if (do_fd) for(int k=0;k<na;++k){ double ae=std::fabs(grad_rev[k]-grad_fd[k]);
    max_abs=std::max(max_abs,ae);
    max_rel=std::max(max_rel, ae/std::max(1e-12,std::fabs(grad_fd[k])));
    norm_err=std::max(norm_err, ae/(atol+rtol*std::fabs(grad_fd[k]))); }

  return Rcpp::List::create(
    Rcpp::Named("F")=F0, Rcpp::Named("F_replay")=Frep,
    Rcpp::Named("grad_rev")=Rcpp::wrap(grad_rev), Rcpp::Named("grad_fd")=Rcpp::wrap(grad_fd),
    Rcpp::Named("max_rel_err")=max_rel, Rcpp::Named("max_abs_err")=max_abs,
    Rcpp::Named("norm_err")=norm_err,
    Rcpp::Named("wall_record_ms")=wall_record, Rcpp::Named("wall_fwd_ms")=wall_fwd,
    Rcpp::Named("wall_rev_ms")=wall_rev, Rcpp::Named("wall_fd_ms")=wall_fd,
    Rcpp::Named("soil_steps")=(double)soil_steps, Rcpp::Named("n_macro")=(double)(macro_times.size()-1),
    Rcpp::Named("rev_over_fwd")=wall_rev/std::max(1e-9,wall_fwd));
}
