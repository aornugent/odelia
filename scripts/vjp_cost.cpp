// Cost of recording a block on an XAD tape and sweeping it in reverse, relative
// to evaluating the same block once in plain double.
//
// Three arms on one callable, interleaved in a single loop so all see the same
// machine load:
//   arm D  one evaluation of the block at S = double
//   arm R  one odelia::ode::vector_jacobian_product, tape constructed per call
//   arm T  the same product on a tape the caller owns and reuses across calls
// The measurement is the ratio R/D against T/D. Absolute times belong to this
// machine; only a ratio taken in one session transfers.
//
// The block stands in for one cohort's rate chain at one Runge-Kutta stage:
// `n_state` ODE states, `n_knot` light-interpolant knot values and `n_knot`
// knot slopes read through odelia's cubic Hermite interpolant, `n_layer` soil
// water potentials and `n_trait` scalar parameters, all active inputs; outputs
// are `n_state` rates plus `n_layer` per-layer uptakes. Arithmetic is
// allometric power-law chains plus a quadrature-like reduction over `n_quad`
// points with an integrand at each.
//
// Sizes run at three scales, the middle one sized so arm D lands near 6 us,
// with the outer two about 4x smaller and 4x larger in knot and quadrature
// count. A ratio that changes across the three means the multiplier is not flat
// in block size.
//
// Also reported: the recording size with one output adjoint seeded against all
// of them, which must be equal; and the share of arm R attributable to
// constructing a Tape per call, timed as record/seed/sweep with an empty block.
//
// odelia's headers need C++20 for the element concepts and Rcpp for util::stop,
// and XAD's active tape pointer lives in src/Tape.cpp, so that is compiled in.
//
// Build and run, from the package root:
//   RCPP=$(Rscript -e 'cat(system.file("include", package="Rcpp"))')
//   g++ -O2 -DNDEBUG -std=c++20 -I inst/include -I /usr/share/R/include -I "$RCPP" \
//       scripts/vjp_cost.cpp src/Tape.cpp -o /tmp/p3vjp/vjp_cost -L/usr/lib/R/lib -lR
//   /tmp/p3vjp/vjp_cost

#include <Rcpp.h>  // odelia's util::stop resolves through Rcpp

#include <XAD/XAD.hpp>
#include <odelia/gradient.hpp>
#include <odelia/hermite_interpolator.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

volatile double sink = 0.0;  // consumes arm D's result so it is not dead code

struct Shape {
  std::size_t n_state = 6;
  std::size_t n_layer = 5;
  std::size_t n_trait = 30;
  std::size_t n_knot = 65;
  std::size_t n_quad = 40;

  std::size_t n_input() const { return n_state + 2 * n_knot + n_layer + n_trait; }
  std::size_t n_output() const { return n_state + n_layer; }
};

// One allometric link: a power law of `u` with an exponent carried by a trait.
// The return type is declared because an XAD expression template would hold
// references to this call's temporaries.
template <typename S>
S allometry_of(const S& u, const S& scale, const S& exponent) {
  using std::pow;
  return scale * pow(u + 1e-8, exponent);
}

// The block. `x` is ordered states, knot values, knot slopes, soil potentials,
// traits; `y` is rates then per-layer uptake, written in place.
struct Block {
  Shape shape;
  std::vector<double> knot_positions;  // fixed for a run
  std::vector<double> quad_weights;
  std::vector<double> quad_nodes;

  explicit Block(const Shape& s) : shape(s) {
    knot_positions.resize(shape.n_knot);
    for (std::size_t k = 0; k < shape.n_knot; ++k)
      knot_positions[k] = 20.0 * static_cast<double>(k) / (shape.n_knot - 1);
    quad_weights.resize(shape.n_quad);
    quad_nodes.resize(shape.n_quad);
    for (std::size_t q = 0; q < shape.n_quad; ++q) {
      const double t = (static_cast<double>(q) + 0.5) / shape.n_quad;
      quad_nodes[q] = t;
      quad_weights[q] = 1.0 / shape.n_quad;
    }
  }

  template <typename S>
  void operator()(const std::vector<S>& x, std::vector<S>& y) const {
    using std::pow;
    using std::sqrt;
    const std::size_t nk = shape.n_knot;
    const std::size_t ns = shape.n_state;
    const std::size_t nl = shape.n_layer;

    const S* states = &x[0];
    const S* knot_y = &x[ns];
    const S* knot_m = &x[ns + nk];
    const S* psi = &x[ns + 2 * nk];
    const S* trait = &x[ns + 2 * nk + nl];

    // Height and the leaf-area chain hanging off it.
    const S height = states[0] + 1.0;
    const S area_leaf = allometry_of<S>(height, trait[0], trait[1]);
    const S mass_sapwood = allometry_of<S>(height, trait[2], trait[3]);
    const S mass_bark = allometry_of<S>(height, trait[4], trait[5]);
    const S mass_root = allometry_of<S>(area_leaf, trait[6], trait[7]);

    // The light field, read through the interpolant that the real block uses.
    odelia::interpolator::hermite_interpolator<S> light;
    std::vector<S> yv(knot_y, knot_y + nk), mv(knot_m, knot_m + nk);
    light.init(knot_positions, yv, mv);

    // Quadrature over the crown: the integrand reads the interpolant at an
    // active position and runs one power-law chain per point.
    S assimilation(0.0);
    S transpiration(0.0);
    for (std::size_t q = 0; q < shape.n_quad; ++q) {
      const S z = height * quad_nodes[q];
      S L(0.0), dL(0.0);
      light.value_and_slope(z, L, dL);
      const S alpha = trait[8] * L / (L + trait[9] + 1e-6);
      const S q_area = area_leaf * (1.0 - quad_nodes[q] * quad_nodes[q]);
      // Three links of the allometric chain per point: leaf area to
      // photosynthetic capacity, capacity to conductance, conductance to flux.
      const S capacity = pow(q_area + 1e-8, trait[10]);
      const S conductance = trait[11] * pow(capacity + 1e-8, trait[12]);
      const S flux = pow(conductance + 1e-8, trait[13]) / (1.0 + conductance);
      const S respiring = pow(capacity + 1e-8, trait[14]) * trait[15];
      assimilation += quad_weights[q] * alpha * capacity * (flux - respiring);
      transpiration += quad_weights[q] * alpha * conductance * (1.0 - dL * dL * 1e-6);
    }

    // Soil uptake per layer, driven by root mass and the layer's potential.
    S uptake_total(0.0);
    for (std::size_t l = 0; l < nl; ++l) {
      const S conductance = allometry_of<S>(mass_root, trait[11 + l], trait[16 + l]);
      const S drive = psi[l] - trait[21 + l];
      const S u = conductance * drive / (1.0 + sqrt(drive * drive + 1e-9));
      y[ns + l] = u;
      uptake_total += u;
    }

    // Respiration and turnover close the rate chain.
    const S respiration = trait[26] * (mass_sapwood + mass_bark) + trait[27] * mass_root;
    const S turnover = trait[28] * area_leaf + trait[29] * mass_bark;
    const S net = assimilation * (uptake_total / (transpiration + 1e-6)) - respiration - turnover;

    y[0] = net / (trait[0] * trait[1] + 1e-6);
    for (std::size_t i = 1; i < ns; ++i) {
      y[i] = net * states[i % ns] * (1.0 / (1.0 + static_cast<double>(i))) + turnover;
    }
  }
};

std::vector<double> make_inputs(const Shape& s) {
  std::vector<double> x(s.n_input());
  const std::size_t nk = s.n_knot;
  for (std::size_t i = 0; i < s.n_state; ++i) x[i] = 0.5 + 0.1 * i;
  for (std::size_t k = 0; k < nk; ++k) {
    const double t = static_cast<double>(k) / (nk - 1);
    x[s.n_state + k] = std::exp(-2.0 * (1.0 - t));         // transmittance
    x[s.n_state + nk + k] = 2.0 * std::exp(-2.0 * (1.0 - t));  // dL/dz
  }
  for (std::size_t l = 0; l < s.n_layer; ++l) x[s.n_state + 2 * nk + l] = -0.1 - 0.05 * l;
  for (std::size_t p = 0; p < s.n_trait; ++p)
    x[s.n_state + 2 * nk + s.n_layer + p] = 0.3 + 0.02 * (p % 7);
  return x;
}

using clk = std::chrono::steady_clock;
double seconds(clk::duration d) {
  return std::chrono::duration<double>(d).count();
}

struct Arms {
  double d = 0.0;  // seconds per call, best of the reps
  double r = 0.0;
  double t = 0.0;
};

// All three arms in one loop, alternating, so contention lands on all of them.
Arms time_arms(const Block& block, const std::vector<double>& x, std::size_t n_out,
               std::size_t reps, std::size_t calls, std::vector<double>& d_spread,
               std::vector<double>& r_spread, std::vector<double>& t_spread) {
  std::vector<double> adjoints(n_out, 1.0);
  std::vector<double> input_adjoints(x.size());
  std::vector<double> yd(n_out);
  auto f = [&block](const std::vector<xad::adj<double>::active_type>& xa,
                    std::vector<xad::adj<double>::active_type>& ya) { block(xa, ya); };
  xad::adj<double>::tape_type owned(false);  // arm T reuses this across every call
  Arms best;
  best.d = 1e30;
  best.r = 1e30;
  best.t = 1e30;
  for (std::size_t rep = 0; rep < reps; ++rep) {
    double td = 0.0, tr = 0.0, tt = 0.0;
    for (std::size_t c = 0; c < calls; ++c) {
      auto t0 = clk::now();
      block(x, yd);
      auto t1 = clk::now();
      double acc = 0.0;
      for (double v : yd) acc += v;
      sink = sink + acc;
      auto t2 = clk::now();
      odelia::ode::vector_jacobian_product(x, adjoints, f, input_adjoints);
      auto t3 = clk::now();
      sink = sink + input_adjoints[0];
      auto t4 = clk::now();
      odelia::ode::vector_jacobian_product(owned, x, adjoints, f, input_adjoints);
      auto t5 = clk::now();
      td += seconds(t1 - t0);
      tr += seconds(t3 - t2);
      tt += seconds(t5 - t4);
      sink = sink + input_adjoints[0];
    }
    td /= calls;
    tr /= calls;
    tt /= calls;
    d_spread.push_back(td);
    r_spread.push_back(tr);
    t_spread.push_back(tt);
    if (td < best.d) best.d = td;
    if (tr < best.r) best.r = tr;
    if (tt < best.t) best.t = tt;
  }
  return best;
}

// record/seed/sweep with nothing recorded: the fixed cost of a Tape per call at
// this input count.
double time_empty_tape(const std::vector<double>& x, std::size_t n_out, std::size_t reps,
                       std::size_t calls) {
  std::vector<double> adjoints(n_out, 1.0);
  std::vector<double> input_adjoints(x.size());
  double best = 1e30;
  for (std::size_t rep = 0; rep < reps; ++rep) {
    auto t0 = clk::now();
    for (std::size_t c = 0; c < calls; ++c) {
      odelia::ode::vector_jacobian_product(
          x, adjoints,
          [](const std::vector<xad::adj<double>::active_type>& xa,
             std::vector<xad::adj<double>::active_type>& ya) {
            for (std::size_t i = 0; i < ya.size(); ++i) ya[i] = xa[i % xa.size()];
          },
          input_adjoints);
      sink = sink + input_adjoints[0];
    }
    const double t = seconds(clk::now() - t0) / calls;
    if (t < best) best = t;
  }
  return best;
}

// Constructing and destroying a Tape and nothing else.
double time_tape_only(std::size_t reps, std::size_t calls) {
  using tape_type = xad::adj<double>::tape_type;
  double best = 1e30;
  for (std::size_t rep = 0; rep < reps; ++rep) {
    auto t0 = clk::now();
    for (std::size_t c = 0; c < calls; ++c) {
      tape_type tape;
      tape.deactivate();
      sink = sink + static_cast<double>(tape.getMemory());
    }
    const double t = seconds(clk::now() - t0) / calls;
    if (t < best) best = t;
  }
  return best;
}

double spread_pct(const std::vector<double>& v, double best) {
  double worst = 0.0;
  for (double t : v) if (t > worst) worst = t;
  return 100.0 * (worst - best) / best;
}

Arms run_size(const char* label, const Shape& shape, std::size_t reps, std::size_t calls) {
  Block block(shape);
  const std::vector<double> x = make_inputs(shape);
  std::vector<double> ds, rs, ts;
  Arms a = time_arms(block, x, shape.n_output(), reps, calls, ds, rs, ts);
  std::printf("%-8s inputs %4zu outputs %2zu knots %4zu quad %4zu | "
              "D %8.3f us (+%.0f%%)  R %9.3f us (+%.0f%%)  T %9.3f us (+%.0f%%)"
              "  R/D %6.2f  T/D %6.2f\n",
              label, shape.n_input(), shape.n_output(), shape.n_knot, shape.n_quad,
              a.d * 1e6, spread_pct(ds, a.d), a.r * 1e6, spread_pct(rs, a.r),
              a.t * 1e6, spread_pct(ts, a.t), a.r / a.d, a.t / a.d);
  return a;
}

}  // namespace

int main() {
  Shape mid;
  Shape small = mid;
  small.n_knot = 17;
  small.n_quad = 10;
  Shape large = mid;
  large.n_knot = 260;
  large.n_quad = 160;

  const std::size_t reps = 7;
  std::printf("record-and-sweep multiplier, best of %zu reps\n\n", reps);
  const Arms as = run_size("small", small, reps, 400);
  const Arms am = run_size("mid", mid, reps, 400);
  const Arms al = run_size("large", large, reps, 150);

  // The multiplier on the marginal work, which strips whatever is fixed per
  // call. If this is flat while R/D is not, the difference is fixed cost.
  std::printf("\nmarginal (R2-R1)/(D2-D1): small->mid %.2f  mid->large %.2f\n",
              (am.r - as.r) / (am.d - as.d), (al.r - am.r) / (al.d - am.d));
  std::printf("marginal (T2-T1)/(D2-D1): small->mid %.2f  mid->large %.2f\n",
              (am.t - as.t) / (am.d - as.d), (al.t - am.t) / (al.d - am.d));

  // The recording size must not grow with the number of seeded output adjoints.
  {
    Block block(mid);
    const std::vector<double> x = make_inputs(mid);
    std::vector<double> ia(x.size());
    auto f = [&block](const std::vector<xad::adj<double>::active_type>& xa,
                      std::vector<xad::adj<double>::active_type>& ya) { block(xa, ya); };
    std::vector<double> one(mid.n_output(), 0.0);
    one[0] = 1.0;
    std::vector<double> all(mid.n_output(), 1.0);
    const std::size_t m1 = odelia::ode::vector_jacobian_product(x, one, f, ia);
    const std::size_t mall = odelia::ode::vector_jacobian_product(x, all, f, ia);
    std::printf("\nrecording size: 1 adjoint seeded %zu bytes, %zu seeded %zu bytes\n",
                m1, mid.n_output(), mall);
    // Finite values only: a NaN through the chain would make the recorded and
    // the double arm do different work.
    double s = 0.0;
    for (double v : ia) s += v;
    std::vector<double> y(mid.n_output());
    block(x, y);
    std::printf("finite check: y[0] %g  sum of input adjoints %g\n", y[0], s);
  }

  // Tape construction plus register/seed/sweep with an empty block, at the mid
  // size's input and output counts.
  {
    const std::vector<double> x = make_inputs(mid);
    const double t = time_empty_tape(x, mid.n_output(), 7, 2000);
    std::printf("empty record/seed/sweep at %zu inputs: %.3f us\n", x.size(), t * 1e6);
    std::printf("tape construct/destroy alone: %.3f us\n", time_tape_only(7, 4000) * 1e6);
  }
  return 0;
}
