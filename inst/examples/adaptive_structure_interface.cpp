/* Where an adaptively built structure gets its shape from, compiled on demand by
 * test-ad-adaptive-structure.R (sourceCpp).
 *
 * A node set is chosen by comparing plain values, so building one at an active
 * scalar picks the same nodes as building it plain -- which is what lets a
 * replayed step rebuild an adaptive background without carrying node positions
 * forward from the pass that discovered them. The refiner predicts midpoints with
 * a plain-valued copy of the interpolant for the same reason: predicting at S
 * would re-solve the coefficient band on the active scalar once per refinement
 * pass and record every solve, for nodes and values that come out identical.
 */

// [[Rcpp::depends(Rcpp, odelia)]]
// [[Rcpp::plugins(cpp20)]]

#include <Rcpp.h>
#include <XAD/XAD.hpp>
#include <odelia/interpolator.hpp>
#include <odelia/ode_util.hpp>

#include <cmath>
#include <vector>

using namespace Rcpp;
using RevS = xad::adj<double>::active_type;
using Tape = xad::adj<double>::tape_type;

// Curved enough that the refiner really subdivides, and shaped by a parameter the
// caller can differentiate.
template <class T>
static T decay_with_ripple(double x, const T& k) {
  return exp(-k * x) * (1.0 + 0.5 * sin(4.0 * x));
}

// [[Rcpp::export]]
Rcpp::List adaptive_structure_probe(double k = 1.3, double atol = 1e-6,
                                    double rtol = 1e-6, int nbase = 17,
                                    int max_depth = 16) {
  const std::size_t nb = static_cast<std::size_t>(nbase);
  const std::size_t md = static_cast<std::size_t>(max_depth);
  const double a = 0.0, b = 3.0, read_at = 1.7;

  // Plain build: the nodes a run without a tape would choose.
  odelia::interpolator::basic_interpolator<double> plain;
  plain.construct(
      [&](double x) -> double { return decay_with_ripple<double>(x, k); }, a, b,
      atol, rtol, nb, md);
  const std::vector<double> nodes_plain = plain.get_x();
  const double value_plain = plain.eval(read_at);

  // Active build: k seeded, nothing recorded and nothing replayed.
  std::vector<double> nodes_active;
  double value_active = 0.0, deriv = 0.0, bytes = 0.0, ops = 0.0;
  {
    Tape tape;
    RevS k_active(k);
    tape.registerInput(k_active);
    tape.newRecording();
    odelia::interpolator::basic_interpolator<RevS> active;
    active.construct(
        [&](double x) -> RevS { return decay_with_ripple<RevS>(x, k_active); }, a,
        b, atol, rtol, nb, md);
    nodes_active = active.get_x();
    RevS y = active.eval(read_at);
    bytes = static_cast<double>(tape.getMemory());
    ops = static_cast<double>(tape.getNumOperations());
    tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    tape.computeAdjoints();
    value_active = xad::value(y);
    deriv = xad::derivative(k_active);
  }

  // The same nodes reached the other way: refine plain, then evaluate the target
  // on those nodes at the active scalar. This is the floor the refiner should be
  // close to, since the only extra work it does is one more coefficient solve.
  double bytes_floor = 0.0, deriv_floor = 0.0;
  {
    Tape tape;
    RevS k_active(k);
    tape.registerInput(k_active);
    tape.newRecording();
    std::vector<RevS> y_nodes;
    y_nodes.reserve(nodes_plain.size());
    for (double xi : nodes_plain) {
      y_nodes.push_back(decay_with_ripple<RevS>(xi, k_active));
    }
    odelia::interpolator::basic_interpolator<RevS> on_nodes;
    on_nodes.init(nodes_plain, y_nodes);
    RevS y = on_nodes.eval(read_at);
    bytes_floor = static_cast<double>(tape.getMemory());
    tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    tape.computeAdjoints();
    deriv_floor = xad::derivative(k_active);
  }

  // Central difference on the plain build, so the derivative is checked against
  // something that shares no machinery with the tape.
  const double h = 1e-6 * (std::fabs(k) + 1.0);
  auto value_at = [&](double kk) {
    odelia::interpolator::basic_interpolator<double> f;
    f.construct([&](double x) -> double { return decay_with_ripple<double>(x, kk); },
                a, b, atol, rtol, nb, md);
    return f.eval(read_at);
  };
  const double deriv_fd = (value_at(k + h) - value_at(k - h)) / (2.0 * h);

  std::size_t mismatches = nodes_plain.size() == nodes_active.size() ? 0 : 1;
  double node_max_diff = 0.0;
  if (mismatches == 0) {
    for (std::size_t i = 0; i < nodes_plain.size(); ++i) {
      const double d = std::fabs(nodes_plain[i] - nodes_active[i]);
      if (d != 0.0) ++mismatches;
      node_max_diff = std::max(node_max_diff, d);
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("n_nodes_plain") = static_cast<int>(nodes_plain.size()),
      Rcpp::Named("n_nodes_active") = static_cast<int>(nodes_active.size()),
      Rcpp::Named("node_mismatches") = static_cast<int>(mismatches),
      Rcpp::Named("node_max_diff") = node_max_diff,
      Rcpp::Named("value_plain") = value_plain,
      Rcpp::Named("value_active") = value_active,
      Rcpp::Named("deriv") = deriv,
      Rcpp::Named("deriv_on_plain_nodes") = deriv_floor,
      Rcpp::Named("deriv_fd") = deriv_fd,
      Rcpp::Named("bytes") = bytes,
      Rcpp::Named("bytes_on_plain_nodes") = bytes_floor,
      Rcpp::Named("ops") = ops);
}
