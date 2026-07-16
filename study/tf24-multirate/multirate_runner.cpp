// Multi-rate demonstration for the TF24 soil block coupled to a large, smooth
// canopy block, under the semi-arid rainfall sequence.
//
// Two-way, low-rank coupling (as the numerical-methods consult specifies: the
// small block's rate reads a scalar aggregate of the large block, and vice versa):
//   soil layer i:  d theta_i/dt = (in_i - K(theta_i) - uptake_i)/dz,
//                  uptake_i = t_pot*(0.5 + xbar)*root_i*stress(theta_i),  xbar = mean(x)
//   canopy k:      d x_k/dt = alpha_k*(Sbar - x_k),   Sbar = mean(theta)/theta_sat
// The 5 soil states carry all the rapid, accuracy-limiting structure; the M canopy
// states are slow, smooth trackers. N = 5 + M.
//
// global_run integrates the whole N-state system with one adaptive step size
// (RKCK or RODAS). multirate_run sub-cycles the 5 soil states with their own
// adaptive steps inside each macro step, freezing the canopy aggregate xbar across
// the macro step and advancing the (linear) canopy exactly from the time-averaged
// soil aggregate. Cost of the canopy block thus decouples from the soil step rate.

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <XAD/XAD.hpp>
#include <odelia/ode_solver.hpp>

using namespace odelia;

static long g_rhs_double = 0, g_rhs_twin = 0;
template <typename T> static inline T powT(const T& a, double b) {
  using std::pow; using xad::pow; return pow(a, b);
}

struct P {
  double theta_sat=0.428, K_sat=163.0411, a_psi=1.78e3, n_psi=6.57, theta_res=1e-2;
  double depth_mm=1500.0, a_infil=1.0, b_infil=8.0, theta_wilt=0.12;
  double stiff=1.0, t_pot=4.0, alpha_lo=0.02, alpha_hi=0.10;
};

// ---------- full coupled system (soil 5 + canopy M) ------------------------
template <typename T = double>
class SoilCanopy {
public:
  using value_type = T;
  SoilCanopy(P p_, size_t M_, const std::vector<double>& rain_)
      : p(p_), nL(5), M(M_), rain(rain_) {
    dz = p.depth_mm/double(nL); p_drain = 2*p.n_psi+3;
    root = {0.35,0.28,0.20,0.12,0.05};
    alpha.resize(M);
    for (size_t k=0;k<M;++k) alpha[k] = p.alpha_lo + (p.alpha_hi-p.alpha_lo)*(M>1?double(k)/(M-1):0.0);
    th_init.assign(nL, T(0.30*p.theta_sat));
    double Sbar0 = 0.30;                       // mean(theta)/theta_sat at init
    x_init.assign(M, T(Sbar0));
    t0=0.0; reset();
  }
  size_t ode_size() const { return nL+M; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }
  double rain_at(double t) const { long d=(long)std::floor(t); if(d<0)d=0; if(d>=(long)rain.size())d=rain.size()-1; return rain[d]; }
  T Kf(const T& th) const { T r=th/p.theta_sat; if(r<0)r=T(0.0); return p.stiff*p.K_sat*powT(r,p_drain); }
  T stress(const T& th) const { T s=(th-p.theta_res)/(p.theta_wilt-p.theta_res); if(s<=0)return T(0.0); if(s>=1)return T(1.0); return s*s*(T(3.0)-T(2.0)*s); }
  void compute_rates() {
    if constexpr (std::is_same_v<T,double>) ++g_rhs_double; else ++g_rhs_twin;
    const double rain_t = rain_at(time);
    T xbar = T(0.0); for (size_t k=0;k<M;++k) xbar += x[k]; if (M>0) xbar /= double(M);
    T Tpot = p.t_pot*(T(0.5)+xbar);
    T r0 = th[0]/p.theta_sat; if(r0<0)r0=T(0.0);
    T runoff = T(1.0)-p.a_infil*powT(r0,p.b_infil); if(runoff<0)runoff=T(0.0);
    T infil = rain_t*runoff;
    std::vector<T> out(nL); for(size_t i=0;i<nL;++i) out[i]=Kf(th[i]);
    T thsum=T(0.0);
    for(size_t i=0;i<nL;++i){ T win=(i==0)?infil:out[i-1];
      dth[i]=(win-out[i]-Tpot*root[i]*stress(th[i]))/dz; thsum+=th[i]; }
    T Sbar = thsum/double(nL)/p.theta_sat;
    for(size_t k=0;k<M;++k) dx[k]=alpha[k]*(Sbar-x[k]);
  }
  template <typename It> It set_ode_state(It it,double time_){ time=time_;
    for(size_t i=0;i<nL;++i)th[i]=*it++; for(size_t k=0;k<M;++k)x[k]=*it++; compute_rates(); return it; }
  template <typename It> It set_initial_state(It it,double t0_=0.0){ t0=t0_;
    for(size_t i=0;i<nL;++i)th_init[i]=*it++; for(size_t k=0;k<M;++k)x_init[k]=*it++; return it; }
  template <typename It> It ode_state(It it) const { for(size_t i=0;i<nL;++i)*it++=th[i]; for(size_t k=0;k<M;++k)*it++=x[k]; return it; }
  template <typename It> It ode_initial_state(It it) const { for(size_t i=0;i<nL;++i)*it++=th_init[i]; for(size_t k=0;k<M;++k)*it++=x_init[k]; return it; }
  template <typename It> It ode_rates(It it) const { for(size_t i=0;i<nL;++i)*it++=dth[i]; for(size_t k=0;k<M;++k)*it++=dx[k]; return it; }
  void reset(){ th=th_init; x=x_init; dth.assign(nL,T(0.0)); dx.assign(M,T(0.0)); time=t0; compute_rates(); }
  std::vector<double> pars() const { return { p.stiff, p.t_pot }; }
  template <class S2> using rebind = SoilCanopy<S2>;
  template <typename U> rebind<U> rebind_from() const {
    SoilCanopy<U> s(p,M,rain); std::vector<U> init(nL+M), st(nL+M);
    for(size_t i=0;i<nL;++i){init[i]=U(xad::value(th_init[i])); st[i]=U(xad::value(th[i]));}
    for(size_t k=0;k<M;++k){init[nL+k]=U(xad::value(x_init[k])); st[nL+k]=U(xad::value(x[k]));}
    s.set_initial_state(init.begin(),t0); s.set_ode_state(st.begin(),time); return s;
  }
  size_t n_layers() const { return nL; }
  const std::vector<double>& alphas() const { return alpha; }
private:
  P p; size_t nL,M; std::vector<double> rain;
  double dz,p_drain,t0,time=0.0; std::vector<double> root,alpha;
  std::vector<T> th,x,dth,dx,th_init,x_init;
};

// ---------- soil sub-system for multi-rate: 5 theta + 1 integral J ----------
// xbar is frozen (a plain parameter). J accumulates the time-integral of the soil
// aggregate Sbar = mean(theta)/theta_sat, so its increment over a macro step gives
// the time-average the canopy needs. rebind_from provided so RODAS is available too.
template <typename T = double>
class SoilSub {
public:
  using value_type = T;
  SoilSub(P p_, const std::vector<double>& rain_, double xbar_)
      : p(p_), nL(5), rain(rain_), xbar(xbar_) {
    dz=p.depth_mm/double(nL); p_drain=2*p.n_psi+3; root={0.35,0.28,0.20,0.12,0.05};
    th_init.assign(nL,T(0.30*p.theta_sat)); J_init=T(0.0); t0=0.0; reset();
  }
  size_t ode_size() const { return nL+1; }
  double ode_time() const { return time; }
  double ode_t0() const { return t0; }
  double rain_at(double t) const { long d=(long)std::floor(t); if(d<0)d=0; if(d>=(long)rain.size())d=rain.size()-1; return rain[d]; }
  T Kf(const T& th_) const { T r=th_/p.theta_sat; if(r<0)r=T(0.0); return p.stiff*p.K_sat*powT(r,p_drain); }
  T stress(const T& th_) const { T s=(th_-p.theta_res)/(p.theta_wilt-p.theta_res); if(s<=0)return T(0.0); if(s>=1)return T(1.0); return s*s*(T(3.0)-T(2.0)*s); }
  void compute_rates(){
    if constexpr (std::is_same_v<T,double>) ++g_rhs_double; else ++g_rhs_twin;
    const double rain_t=rain_at(time);
    T Tpot=p.t_pot*(T(0.5)+T(xbar));
    T r0=th[0]/p.theta_sat; if(r0<0)r0=T(0.0);
    T runoff=T(1.0)-p.a_infil*powT(r0,p.b_infil); if(runoff<0)runoff=T(0.0);
    T infil=rain_t*runoff; std::vector<T> out(nL); for(size_t i=0;i<nL;++i)out[i]=Kf(th[i]);
    T thsum=T(0.0);
    for(size_t i=0;i<nL;++i){ T win=(i==0)?infil:out[i-1];
      dth[i]=(win-out[i]-Tpot*root[i]*stress(th[i]))/dz; thsum+=th[i]; }
    dJ = thsum/double(nL)/p.theta_sat;      // dJ/dt = Sbar
  }
  template <typename It> It set_ode_state(It it,double time_){ time=time_;
    for(size_t i=0;i<nL;++i)th[i]=*it++; J=*it++; compute_rates(); return it; }
  template <typename It> It set_initial_state(It it,double t0_=0.0){ t0=t0_;
    for(size_t i=0;i<nL;++i)th_init[i]=*it++; J_init=*it++; return it; }
  template <typename It> It ode_state(It it) const { for(size_t i=0;i<nL;++i)*it++=th[i]; *it++=J; return it; }
  template <typename It> It ode_initial_state(It it) const { for(size_t i=0;i<nL;++i)*it++=th_init[i]; *it++=J_init; return it; }
  template <typename It> It ode_rates(It it) const { for(size_t i=0;i<nL;++i)*it++=dth[i]; *it++=dJ; return it; }
  void reset(){ th=th_init; J=J_init; dth.assign(nL,T(0.0)); dJ=T(0.0); time=t0; compute_rates(); }
  std::vector<double> pars() const { return { p.stiff, xbar }; }
  template <class S2> using rebind = SoilSub<S2>;
  template <typename U> rebind<U> rebind_from() const {
    SoilSub<U> s(p,rain,xbar); std::vector<U> init(nL+1),st(nL+1);
    for(size_t i=0;i<nL;++i){init[i]=U(xad::value(th_init[i])); st[i]=U(xad::value(th[i]));}
    init[nL]=U(xad::value(J_init)); st[nL]=U(xad::value(J));
    s.set_initial_state(init.begin(),t0); s.set_ode_state(st.begin(),time); return s;
  }
private:
  P p; size_t nL; std::vector<double> rain; double xbar;
  double dz,p_drain,t0,time=0.0; std::vector<double> root;
  std::vector<T> th,dth,th_init; T J,dJ,J_init;
};

static ode::Method pick(const std::string& m){ return (m=="rodas")?ode::Method::rodas:ode::Method::rkck; }

// ---------- global run: one adaptive step size over the whole N-state system --
// [[Rcpp::export]]
Rcpp::List global_run(std::vector<double> rain, std::vector<double> out_times,
                      int M, double tol_rel, double tol_abs, std::string method,
                      double stiff, double t_pot) {
  P p; p.stiff=stiff; p.t_pot=t_pot;
  using Sys=SoilCanopy<double>; Sys sys(p,(size_t)M,rain);
  ode::OdeControl ctrl(tol_abs,tol_rel,1.0,0.0,1e-10,1e6,1e-4);
  ode::Solver<Sys> solver(sys,ctrl,pick(method)); solver.set_collect(true);
  g_rhs_double=0; g_rhs_twin=0;
  auto t0=std::chrono::high_resolution_clock::now();
  solver.advance_adaptive(out_times);
  auto t1=std::chrono::high_resolution_clock::now();
  double wall=std::chrono::duration<double,std::milli>(t1-t0).count();
  auto hist=solver.get_history(); size_t nL=sys.n_layers(), ng=hist.size();
  Rcpp::NumericMatrix theta(ng,nL); std::vector<double> s(nL+M);
  Rcpp::NumericVector xbar(ng);
  std::vector<double> xf(M);
  for(size_t i=0;i<ng;++i){ hist[i].ode_state(s.begin());
    for(size_t j=0;j<nL;++j)theta(i,j)=s[j];
    double xb=0; for(int k=0;k<M;++k)xb+=s[nL+k]; xbar[i]=(M>0)?xb/M:0.0; }
  if (M>0){ hist[ng-1].ode_state(s.begin()); for(int k=0;k<M;++k)xf[k]=s[nL+k]; }
  return Rcpp::List::create(
    Rcpp::Named("method")=method, Rcpp::Named("M")=M,
    Rcpp::Named("n_steps")=(double)solver.times().size(),
    Rcpp::Named("wall_ms")=wall, Rcpp::Named("rhs_double")=(double)g_rhs_double,
    Rcpp::Named("rhs_twin")=(double)g_rhs_twin,
    Rcpp::Named("theta")=theta, Rcpp::Named("xbar")=xbar,
    Rcpp::Named("x_final")=Rcpp::wrap(xf));
}

// ---------- multi-rate run: soil sub-cycled, canopy macro-stepped ------------
// macro_times: the macro grid (e.g. every H days). Between consecutive macro
// times the soil (5+J) is integrated adaptively with xbar frozen; the canopy is
// then advanced exactly using S_avg = dJ/H. soil_method may be "rkck" or "rodas".
// [[Rcpp::export]]
Rcpp::List multirate_run(std::vector<double> rain, std::vector<double> macro_times,
                         int M, double tol_rel, double tol_abs, std::string soil_method,
                         double stiff, double t_pot,
                         double alpha_lo, double alpha_hi) {
  P p; p.stiff=stiff; p.t_pot=t_pot; p.alpha_lo=alpha_lo; p.alpha_hi=alpha_hi;
  const size_t nL=5;
  std::vector<double> alpha(M);
  for(int k=0;k<M;++k) alpha[k]=alpha_lo+(alpha_hi-alpha_lo)*((M>1)?double(k)/(M-1):0.0);
  // state
  std::vector<double> th(nL, 0.30*p.theta_sat);
  std::vector<double> x(M, 0.30);            // canopy init = Sbar0
  // Cap the soil sub-step so a fresh per-macro solve cannot leap across a storm
  // onset inside the macro window (an explicit overshoot of the fast drainage).
  double hmax = std::min(0.5, 0.5 * (macro_times.size() > 1 ? macro_times[1]-macro_times[0] : 1.0));
  ode::OdeControl ctrl(tol_abs,tol_rel,1.0,0.0,1e-10,hmax,1e-4);

  size_t ng=macro_times.size();
  Rcpp::NumericMatrix theta_tr(ng,nL); Rcpp::NumericVector xbar_tr(ng);
  for(size_t j=0;j<nL;++j) theta_tr(0,j)=th[j];
  { double xb=0; for(int k=0;k<M;++k)xb+=x[k]; xbar_tr[0]=(M>0)?xb/M:0.0; }

  g_rhs_double=0; g_rhs_twin=0;
  long soil_steps=0;
  auto t0=std::chrono::high_resolution_clock::now();
  for(size_t m=0;m+1<ng;++m){
    double ta=macro_times[m], tb=macro_times[m+1], H=tb-ta;
    double xbar=0; for(int k=0;k<M;++k)xbar+=x[k]; if(M>0)xbar/=M;   // frozen aggregate
    // sub-cycle soil (5 theta + J) over [ta,tb]
    SoilSub<double> sub(p, rain, xbar);
    std::vector<double> y0(nL+1); for(size_t i=0;i<nL;++i)y0[i]=th[i]; y0[nL]=0.0; // J restart at 0
    ode::Solver<SoilSub<double>> ssolver(sub, ctrl, pick(soil_method));
    ssolver.set_collect(false);
    ssolver.set_state(y0, ta);
    std::vector<double> grid{ta, tb};
    ssolver.advance_adaptive(grid);
    soil_steps += (long)ssolver.times().size();
    auto st = ssolver.state();
    for(size_t i=0;i<nL;++i) th[i]=st[i];
    double S_avg = st[nL]/H;                 // time-average of Sbar over [ta,tb]
    // advance canopy exactly: dx/dt = alpha*(S_avg - x) with S_avg constant
    for(int k=0;k<M;++k) x[k] = S_avg + (x[k]-S_avg)*std::exp(-alpha[k]*H);
    // record
    for(size_t j=0;j<nL;++j) theta_tr(m+1,j)=th[j];
    double xb=0; for(int k=0;k<M;++k)xb+=x[k]; xbar_tr[m+1]=(M>0)?xb/M:0.0;
  }
  auto t1=std::chrono::high_resolution_clock::now();
  double wall=std::chrono::duration<double,std::milli>(t1-t0).count();
  return Rcpp::List::create(
    Rcpp::Named("M")=M, Rcpp::Named("n_macro")=(double)(ng-1),
    Rcpp::Named("soil_steps")=(double)soil_steps, Rcpp::Named("wall_ms")=wall,
    Rcpp::Named("rhs_double")=(double)g_rhs_double,
    Rcpp::Named("theta")=theta_tr, Rcpp::Named("xbar")=xbar_tr,
    Rcpp::Named("x_final")=Rcpp::wrap(x));
}
