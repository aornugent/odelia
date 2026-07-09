#ifndef LORENZ_SYSTEM_HPP_
#define LORENZ_SYSTEM_HPP_

#include <odelia/ode_solver.hpp>
#include <XAD/XAD.hpp>

using namespace odelia;

// Templated Lorenz system - works with double or XAD active types
template <typename T = double>
class LorenzSystem {
public:
  using value_type = T; 
  
  LorenzSystem(T sigma_, T R_, T b_)
    : y0_init(1.0), y1_init(1.0), y2_init(1.0),
      t0(0.0),
      sigma(sigma_), R(R_), b(b_),
      dy0dt(0.0), dy1dt(0.0), dy2dt(0.0) {
    reset();  // initialises state & time
  }

  // Lift this system's configuration into a copy templated on a different scalar
  // (double -> active): the driver builds the active system from the plain one
  // generically. Only values cross, so the new system carries no tape identity;
  // the driver seeds its active inputs afterwards via set_param / set_ic.
  template <class S2> using rebind = LorenzSystem<S2>;

  template <class S2>
  rebind<S2> rebind_from() const {
    rebind<S2> out(xad::value(sigma), xad::value(R), xad::value(b));
    const double ic[] = {xad::value(y0_init), xad::value(y1_init), xad::value(y2_init)};
    out.set_initial_state(ic, t0);
    return out;
  }

  // ODE interface
  size_t ode_size() const { return ode_dimension; }

  double ode_time() const { return time; }

  double ode_t0() const { return t0; }

  template <typename Iterator>
  Iterator set_ode_state(Iterator it, double time_) {
    time = time_;
    
    y0 = *it++;
    y1 = *it++;
    y2 = *it++;
    
    compute_rates();
    return it;
  }

  void compute_rates() {
    dy0dt = sigma * (y1 - y0);
    dy1dt = R * y0 - y1 - y0 * y2;
    dy2dt = -b * y2 + y0 * y1;
  }

  template <typename Iterator>
  Iterator set_initial_state(Iterator it, double t0_ = 0.0) {
    t0 = t0_;
    y0_init = *it++;
    y1_init = *it++;
    y2_init = *it++;
    return it;
  }

  template <typename Iterator>
  Iterator set_params(Iterator it) {
    sigma = *it++;
    R = *it++;
    b = *it++;
    return it;
  }

  // AD input contract: seed one active parameter or initial-state value by index.
  // Params: 0=sigma, 1=R, 2=b. ICs: 0=y0, 1=y1, 2=y2.
  void set_param(int i, T v) {
    switch (i) {
      case 0: sigma = v; break;
      case 1: R     = v; break;
      case 2: b     = v; break;
      default: util::stop("LorenzSystem::set_param: index out of range");
    }
  }
  void set_ic(int j, T v) {
    switch (j) {
      case 0: y0_init = v; break;
      case 1: y1_init = v; break;
      case 2: y2_init = v; break;
      default: util::stop("LorenzSystem::set_ic: index out of range");
    }
  }

  template <typename Iterator>
  Iterator ode_state(Iterator it) const {
    *it++ = y0;
    *it++ = y1;
    *it++ = y2;
    return it;
  }

  template <typename Iterator>
  Iterator ode_initial_state(Iterator it) const {
    *it++ = y0_init;
    *it++ = y1_init;
    *it++ = y2_init;
    return it;
  }

  template <typename Iterator>
  Iterator ode_rates(Iterator it) const {
    *it++ = dy0dt;
    *it++ = dy1dt;
    *it++ = dy2dt;
    return it;
  }

  std::vector<double> record_step() const {
    std::vector<double> ret;
    ret.reserve(7);
    
    ret.push_back(time);
    ret.push_back(xad::value(y0));
    ret.push_back(xad::value(y1));
    ret.push_back(xad::value(y2));
    ret.push_back(xad::value(dy0dt));
    ret.push_back(xad::value(dy1dt));
    ret.push_back(xad::value(dy2dt));
    
    return ret;
  }

  std::vector<double> pars() const {
    std::vector<double> ret;
    ret.push_back(xad::value(sigma));
    ret.push_back(xad::value(R));
    ret.push_back(xad::value(b));
    return ret;
  }

  void reset() {
    y0 = y0_init;
    y1 = y1_init;
    y2 = y2_init;
    time = t0;
    compute_rates();
  }

private:
  static const int ode_dimension = 3;

  T y0_init, y1_init, y2_init;
  double t0; 

  T sigma, R, b; 
  T y0, y1, y2; 
  T dy0dt, dy1dt, dy2dt;

  double time;   
};

#endif
