/* Exercises odelia::implicit_value -- a scalar inner solve differentiated by the
 * implicit function theorem, not by recording the iteration -- compiled on demand
 * by test-ad-implicit-node.R (sourceCpp).
 *
 * Residual F(y; a, b) = y - a cos(y) - b = 0, a transcendental root with no closed
 * form (so differentiating through the Newton iteration is exactly what the node
 * avoids). dy/da = cos(y)/(1 + a sin y), dy/db = 1/(1 + a sin y). The demo
 * differentiates g = y^2 and returns the reverse gradient vs analytic, the IFT
 * partials vs a re-solve finite difference, the forward-vs-reverse dot-product
 * oracle, and -- on a maximiser with dF/dy < 0 -- the denom_sign guard firing when
 * the declared sign is wrong. F reads the active parameters from scope (the common
 * Strategy case), so no input vector is marshalled.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/implicit_node.hpp>

#include <cmath>
#include <vector>

using namespace Rcpp;

// Off-tape Newton root of F(y; a, b) = 0 from double parameter values.
static double solve_root(double a, double b) {
  double y = 1.0;
  for (int i = 0; i < 100; ++i) {
    const double f = y - a * std::cos(y) - b;
    const double dy = f / (1.0 + a * std::sin(y));  // F_y = 1 + a sin(y)
    y -= dy;
    if (std::abs(dy) < 1e-15) break;
  }
  return y;
}

// An N_p*-shaped inner solve: a concave objective W(q; a) = a q - q^3/3 whose argmax
// q* = sqrt(a) is found by golden-section (a branchy, non-Newton iterator),
// registered through its stationarity residual G(q) = dW/dq = a - q^2 with a
// NEGATIVE denominator dG/dq = -2q < 0. dq*/da = -(dG/da)/(dG/dq) = 1/(2 sqrt(a)).
static double golden_argmax(double a) {
  auto W = [&](double q) { return a * q - q * q * q / 3.0; };
  double lo = 1e-3, hi = 10.0;
  const double gr = 0.6180339887498949;
  double c = hi - gr * (hi - lo), d = lo + gr * (hi - lo);
  for (int i = 0; i < 300; ++i) {
    if (W(c) > W(d)) hi = d; else lo = c;
    c = hi - gr * (hi - lo);
    d = lo + gr * (hi - lo);
    if (hi - lo < 1e-14) break;
  }
  return 0.5 * (lo + hi);
}

// [[Rcpp::export]]
Rcpp::List implicit_node_optimum_demo(double a = 2.0) {
  using AD = xad::adj<double>::active_type;
  xad::adj<double>::tape_type tape;
  AD A = a;
  tape.registerInput(A);
  tape.newRecording();
  const double qstar_d = golden_argmax(a);
  // Stationarity residual G(q) = a - q^2, reading the active A from scope; dG/dq < 0
  // (a maximiser), so we declare the negative branch and it verifies.
  AD q = odelia::implicit_value<AD>(qstar_d, [&](AD qq) { return A - qq * qq; },
                                    odelia::denom_sign::negative);
  tape.registerOutput(q);
  xad::derivative(q) = 1.0;
  tape.computeAdjoints();
  const double qstar = xad::value(q);
  const double dq_da = xad::derivative(A);

  // The guard fires when the declared denominator sign is wrong: this maximiser has
  // dG/dq < 0, so declaring positive must stop, not return. The check runs before
  // any recording, so reusing the (already-swept) active tape is fine.
  bool sign_assert_fired = false;
  try {
    odelia::implicit_value<AD>(qstar_d, [&](AD qq) { return A - qq * qq; },
                               odelia::denom_sign::positive);
  } catch (...) {
    sign_assert_fired = true;
  }

  return Rcpp::List::create(
      Rcpp::Named("q_star") = qstar,
      Rcpp::Named("q_star_analytic") = std::sqrt(a),
      Rcpp::Named("dq_da") = dq_da,
      // The IFT sensitivity is exact given the operating point: 1/(2 q*) at the q*
      // the solver returned (separating IFT correctness from golden-section's ~1e-7
      // location accuracy on a flat maximum).
      Rcpp::Named("dq_da_ift") = 1.0 / (2.0 * qstar),
      Rcpp::Named("sign_assert_fired") = sign_assert_fired);
}

// [[Rcpp::export]]
Rcpp::List implicit_node_demo(double a = 0.5, double b = 1.0,
                              double va = 0.4, double vb = -0.9) {
  using AD = xad::adj<double>::active_type;
  using FAD = xad::fwd<double>::active_type;
  const double y_star = solve_root(a, b);

  // Reverse: g = y^2, gradient w.r.t. (a, b). F reads the active A, B from scope.
  xad::adj<double>::tape_type tape;
  AD A = a, B = b;
  tape.registerInput(A);
  tape.registerInput(B);
  tape.newRecording();
  AD y = odelia::implicit_value<AD>(
      y_star, [&](AD yy) { return yy - A * cos(yy) - B; }, odelia::denom_sign::positive);
  AD g = y * y;
  tape.registerOutput(g);
  xad::derivative(g) = 1.0;
  tape.computeAdjoints();
  const double grad_a = xad::derivative(A), grad_b = xad::derivative(B);

  // Forward: directional derivative of g along (va, vb).
  FAD Af = a, Bf = b;
  xad::derivative(Af) = va;
  xad::derivative(Bf) = vb;
  FAD yf = odelia::implicit_value<FAD>(
      y_star, [&](FAD yy) { return yy - Af * cos(yy) - Bf; });
  FAD gf = yf * yf;
  const double jvp = xad::derivative(gf);

  // Plain double: no derivatives, just the root passed straight through.
  double yd = odelia::implicit_value<double>(
      y_star, [&](double yy) { return yy - a * std::cos(yy) - b; });

  // Analytic IFT partials, and a re-solve finite difference of y*(a).
  const double dyda = std::cos(y_star) / (1.0 + a * std::sin(y_star));
  const double dydb = 1.0 / (1.0 + a * std::sin(y_star));
  const double h = 1e-6;
  const double fd_dyda = (solve_root(a + h, b) - solve_root(a - h, b)) / (2 * h);

  return Rcpp::List::create(
      Rcpp::Named("y_star") = y_star,
      Rcpp::Named("y_double") = yd,
      Rcpp::Named("grad_a") = grad_a,
      Rcpp::Named("grad_b") = grad_b,
      Rcpp::Named("grad_a_analytic") = 2.0 * y_star * dyda,
      Rcpp::Named("grad_b_analytic") = 2.0 * y_star * dydb,
      Rcpp::Named("dyda_analytic") = dyda,
      Rcpp::Named("dyda_fd") = fd_dyda,
      Rcpp::Named("jvp") = jvp,
      Rcpp::Named("dot_v_grad") = grad_a * va + grad_b * vb);
}
