// [[Rcpp::plugins(cpp20)]]
// Demonstrates odelia::ad::directional_derivative: a forward-over-reverse
// directional derivative that stays active on an outer reverse tape.
//
// The test function mirrors the shape of plant's growth rate: g(h) is smooth
// arithmetic with a clamp (`if (g < 0) g = 0`, a plant cannot shrink). We take
// dg/dh by forward mode over the reverse type, then differentiate THAT with
// respect to a parameter b0 by the outer reverse pass -- the exact mixed partial
// d(dg/dh)/db0. We compare against the analytic value and against the on-tape
// finite-difference-of-a-finite-difference that this helper is meant to replace.
#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/directional_derivative.hpp>
#include <cmath>

using Areal = xad::adj<double>::active_type;  // outer reverse scalar (over b0)

// g(h; b0, b1, b2, C), templated on the scalar type. `b*` are promoted into the
// same type as `h` by the caller; C is a genuine double constant.
template <class T>
static T growth(T h, T b0, T b1, T b2, double C) {
  T g = h * (b0 - b1 * log(h) - b2 * C);
  if (g < 0.0) g = T(0.0);
  return g;
}

// [[Rcpp::export]]
Rcpp::List directional_derivative_demo(double h0 = 5.0, double b0 = 0.059,
                                       double b1 = 0.012, double b2 = 0.00041,
                                       double C = 0.0, double delta = 1e-4) {
  xad::Tape<double> tape;
  Areal b0a = b0;
  tape.registerInput(b0a);
  tape.newRecording();

  // dg/dh, exact, still active on the outer tape over b0. The lambda lifts the
  // parameters (constant w.r.t. the h direction) into the tangent layer.
  const Areal b1a = b1, b2a = b2;
  Areal dgdh = odelia::ad::directional_derivative(Areal(h0),
    [&](odelia::ad::tangent_of<Areal> h) {
      using odelia::ad::constant;
      return growth(h, constant(b0a), constant(b1a), constant(b2a), C);
    });

  tape.registerOutput(dgdh);
  derivative(dgdh) = 1.0;
  tape.computeAdjoints();

  const double dgdh_value = xad::value(xad::value(dgdh));  // strip both layers
  const double d2_ad = derivative(b0a);                    // d(dg/dh)/db0

  // Analytic references. g = h*(b0 - b1 ln h - b2 C):
  //   dg/dh          = b0 - b1 ln h - b1 - b2 C
  //   d(dg/dh)/db0   = 1        (unclamped)  or 0 (clamped)
  const bool clamped = growth<double>(h0, b0, b1, b2, C) <= 0.0;
  const double dgdh_analytic =
      clamped ? 0.0 : (b0 - b1 * std::log(h0) - b1 - b2 * C);
  const double d2_analytic = clamped ? 0.0 : 1.0;

  // The approach this helper replaces: finite-difference dg/dh in h, then
  // finite-difference THAT in b0. Reported so the test can show it is what breaks
  // near the clamp (here computed in double; on the tape the second stage is what
  // the reverse pass does to the recorded stencil).
  auto dgdh_fd = [&](double bb0) {
    const double e = 1e-6;
    return (growth<double>(h0, bb0, b1, b2, C) -
            growth<double>(h0 - e, bb0, b1, b2, C)) / e;
  };
  const double d2_fd_of_fd = (dgdh_fd(b0 + delta) - dgdh_fd(b0 - delta)) / (2 * delta);

  return Rcpp::List::create(
      Rcpp::Named("dgdh") = dgdh_value,
      Rcpp::Named("dgdh_analytic") = dgdh_analytic,
      Rcpp::Named("d2_ad") = d2_ad,
      Rcpp::Named("d2_analytic") = d2_analytic,
      Rcpp::Named("d2_fd_of_fd") = d2_fd_of_fd,
      Rcpp::Named("clamped") = clamped);
}
