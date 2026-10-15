// Exactness checks for the RKCK stepper, using a closed-form propagator.
//
// The failure mode this exists for is a stepper that is wrong and still returns
// a finite, plausible number. A trajectory compared against a tighter tolerance
// (as in r_free.cpp) cannot see a mistyped tableau entry or a stage wired to the
// wrong predecessor: the answer stays convergent, just to the wrong method. What
// is needed is an oracle that is exact, and for an explicit Runge-Kutta step
// there is one.
//
// For a linear system y' = A(t) y an explicit RK step is exactly linear in the
// state, y_{n+1} = M y_n, and M follows from the tableau alone:
//
//     K_1 = A(t_n)
//     K_i = A(t_n + c_i h) * (I + h * sum_{l<i} a_il K_l)
//     M   = I + h * sum_i b_i K_i
//     E   = h * sum_i (b_i - b*_i) K_i          (the error estimate)
//
// No quadrature error, no truncation, no reference implementation. Stepping the
// n basis vectors through Step<> measures M column by column, and the two must
// agree entry by entry. They agree *bitwise* -- see the note on association
// below -- so there is no tolerance to choose and no tolerance to tune.
//
// The tableau below is written from the literature (Cash & Karp 1990), not read
// off ode_step.hpp, so a transcription error in either one shows up as a
// disagreement. Note which entries are zero: it is the fifth-order weights b_2
// and b_5, not the abscissae. Every abscissa c = (0, 1/5, 3/10, 3/5, 1, 7/8) is
// distinct and only the first is zero.
//
// WHY THE SYSTEM IS TIME-VARYING. For y' = A y with constant A, each stage is
// k_i = A(y_n + h sum a_il k_l): the stage times t_n + c_i h are computed and
// then never used. An abscissa could be arbitrarily wrong and no autonomous test
// could tell. test_abscissae_need_a_time_varying_system() measures exactly that
// rather than asserting it.
//
// Scope is forward only. The transpose half -- ybar_n = M^T ybar_{n+1} -- would
// be the same M compared against a reverse pass, and needs nothing here that is
// not already built.
//
// This file is compiled with -ffp-contract=off (see the Makefile): it holds two
// spellings of one real-number expression to bitwise agreement, and contracting
// a*b+c into an FMA in one spelling but not the other breaks that on rounding
// rather than on a defect.

#include <odelia/ode_interface.hpp>
#include <odelia/ode_solver.hpp>
#include <odelia/ode_step.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) {
    ++failures;
  }
}

// Distance in representable doubles. Only used to report *how far* a failure
// missed by; every assertion below is exact equality.
long ulp_dist(double a, double b) {
  if (a == b) {
    return 0;
  }
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return -1;
  }
  std::int64_t ia, ib;
  std::memcpy(&ia, &a, sizeof(ia));
  std::memcpy(&ib, &b, sizeof(ib));
  if (ia < 0) { ia = INT64_MIN - ia; }
  if (ib < 0) { ib = INT64_MIN - ib; }
  return ia > ib ? ia - ib : ib - ia;
}

// --- the system -------------------------------------------------------------
//
// Three states, so the entry-by-entry comparison is exhaustive (9 cells) and a
// failure localises to a cell. A(t) = P + t Q; Q = 0 gives the autonomous twin
// through the identical code path, so the only thing that differs between the
// two is whether A actually depends on t.

constexpr std::size_t N = 3;
using Mat = std::array<double, N * N>;

// Counting derivative evaluations is how the rejection test proves it is not
// vacuous. A free counter rather than a member because Solver copies the system.
long derivs_calls = 0;

struct LinearSystem {
  using value_type = double;
  Mat P{}, Q{};
  double t = 0.0;
  std::vector<double> y = std::vector<double>(N, 0.0);

  std::size_t ode_size() const { return N; }
  double ode_time() const { return t; }

  Mat A(double time) const {
    Mat out{};
    for (std::size_t i = 0; i < N * N; ++i) {
      out[i] = P[i] + time * Q[i];
    }
    return out;
  }

  template <class It> It set_ode_state(It it, double time) {
    for (std::size_t i = 0; i < N; ++i) { y[i] = *it++; }
    t = time;
    return it;
  }
  template <class It> It ode_state(It it) const {
    for (std::size_t i = 0; i < N; ++i) { *it++ = y[i]; }
    return it;
  }
  template <class It> It ode_rates(It it) const {
    const Mat a = A(t);
    ++derivs_calls;
    for (std::size_t i = 0; i < N; ++i) {
      double acc = 0.0;
      for (std::size_t l = 0; l < N; ++l) { acc += a[i * N + l] * y[l]; }
      *it++ = acc;
    }
    return it;
  }
  template <class It> It ode_aux(It it) const { return it; }
};

LinearSystem make_system(bool time_varying) {
  LinearSystem s;
  s.P = {-0.7,  0.3, -0.2,
          0.5, -1.3,  0.9,
         -0.4,  0.6, -0.8};
  if (time_varying) {
    s.Q = { 0.9, -0.5,  0.4,
           -0.3,  0.7, -0.6,
            0.2, -0.8,  1.1};
  }
  s.y = {1.0, -0.5, 0.25};
  return s;
}

// --- the tableau ------------------------------------------------------------
//
// Cash-Karp 4(5), from the literature. b is the fifth-order weight set the step
// advances on, bstar the fourth-order set the error estimate differences against.
// The zeros are stated explicitly rather than omitted, so that "b_2 and b_5 are
// zero" is a claim this file tests rather than a claim it assumes.

struct Tableau {
  double c[6];
  double a[6][6];
  double b[6];
  double bstar[6];
};

Tableau cash_karp() {
  Tableau T{};

  T.c[0] = 0.0;
  T.c[1] = 1.0 / 5.0;
  T.c[2] = 3.0 / 10.0;
  T.c[3] = 3.0 / 5.0;
  T.c[4] = 1.0;
  T.c[5] = 7.0 / 8.0;

  T.a[1][0] = 1.0 / 5.0;

  T.a[2][0] = 3.0 / 40.0;
  T.a[2][1] = 9.0 / 40.0;

  T.a[3][0] = 3.0 / 10.0;
  T.a[3][1] = -9.0 / 10.0;
  T.a[3][2] = 6.0 / 5.0;

  T.a[4][0] = -11.0 / 54.0;
  T.a[4][1] = 5.0 / 2.0;
  T.a[4][2] = -70.0 / 27.0;
  T.a[4][3] = 35.0 / 27.0;

  T.a[5][0] = 1631.0 / 55296.0;
  T.a[5][1] = 175.0 / 512.0;
  T.a[5][2] = 575.0 / 13824.0;
  T.a[5][3] = 44275.0 / 110592.0;
  T.a[5][4] = 253.0 / 4096.0;

  T.b[0] = 37.0 / 378.0;
  T.b[1] = 0.0;
  T.b[2] = 250.0 / 621.0;
  T.b[3] = 125.0 / 594.0;
  T.b[4] = 0.0;
  T.b[5] = 512.0 / 1771.0;

  T.bstar[0] = 2825.0 / 27648.0;
  T.bstar[1] = 0.0;
  T.bstar[2] = 18575.0 / 48384.0;
  T.bstar[3] = 13525.0 / 55296.0;
  T.bstar[4] = 277.0 / 14336.0;
  T.bstar[5] = 0.25;

  return T;
}

// --- the closed-form propagator ---------------------------------------------

Mat identity() {
  Mat I{};
  for (std::size_t i = 0; i < N; ++i) { I[i * N + i] = 1.0; }
  return I;
}

Mat matmul(const Mat& X, const Mat& Y) {
  Mat Z{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (std::size_t l = 0; l < N; ++l) { acc += X[i * N + l] * Y[l * N + j]; }
      Z[i * N + j] = acc;
    }
  }
  return Z;
}

// M and E from the tableau. K[i] is the matrix with k_i = K[i] y_n.
//
// The floating-point association mirrors ode_step.hpp's -- notably stage 2,
// which the header writes `y[i] + b21 * h * k1[i]`, i.e. (a21*h)*K_1 rather than
// h*(a21*K_1). That is the only place the two shapes differ, and it is worth
// mirroring: with it the agreement is bitwise over every input tried; without
// it, 31% of random inputs disagree in the last bit and the check would need a
// tolerance. Association is not a correctness property of an RK method, so if
// the header's changes, change this and keep the equality exact -- do not
// loosen the comparison.
void propagator(const LinearSystem& sys, const Tableau& T, double t0, double h,
                Mat& M, Mat& E) {
  const Mat I = identity();
  Mat K[6];

  for (std::size_t i = 0; i < 6; ++i) {
    Mat S;
    if (i == 0) {
      S = I;                       // stage 1 is evaluated at y_n itself
    } else if (i == 1) {
      const double ah = T.a[1][0] * h;
      for (std::size_t e = 0; e < N * N; ++e) { S[e] = I[e] + ah * K[0][e]; }
    } else {
      for (std::size_t e = 0; e < N * N; ++e) {
        double acc = 0.0;
        for (std::size_t l = 0; l < i; ++l) { acc += T.a[i][l] * K[l][e]; }
        S[e] = I[e] + h * acc;
      }
    }
    // Stage 1 is the derivative the solver already holds, taken at t_n exactly;
    // the stepper does not form t_n + c_1 h for it.
    const double ti = (i == 0) ? t0 : t0 + T.c[i] * h;
    K[i] = matmul(sys.A(ti), S);
  }

  for (std::size_t e = 0; e < N * N; ++e) {
    double macc = 0.0, eacc = 0.0;
    for (std::size_t l = 0; l < 6; ++l) {
      macc += T.b[l] * K[l][e];
      eacc += (T.b[l] - T.bstar[l]) * K[l][e];
    }
    M[e] = I[e] + h * macc;
    E[e] = h * eacc;
  }
}

// M and E measured by pushing each basis vector through one Step<>.
void measure(const LinearSystem& proto, double t0, double h, Mat& M, Mat& E) {
  for (std::size_t j = 0; j < N; ++j) {
    LinearSystem sys = proto;
    std::vector<double> y(N, 0.0);
    y[j] = 1.0;
    std::vector<double> dydt_in(N), yerr(N), dydt_out(N);
    odelia::ode::derivs(sys, y, dydt_in, t0);

    odelia::ode::Step<LinearSystem> stepper;
    stepper.resize(N);
    stepper.step(sys, t0, h, y, yerr, dydt_in, dydt_out);

    for (std::size_t i = 0; i < N; ++i) {
      M[i * N + j] = y[i];
      E[i * N + j] = yerr[i];
    }
  }
}

long worst_ulp(const Mat& X, const Mat& Y) {
  long worst = 0;
  for (std::size_t e = 0; e < N * N; ++e) {
    const long d = ulp_dist(X[e], Y[e]);
    if (d < 0) { return -1; }
    if (d > worst) { worst = d; }
  }
  return worst;
}

double max_abs_diff(const Mat& X, const Mat& Y) {
  double worst = 0.0;
  for (std::size_t e = 0; e < N * N; ++e) {
    worst = std::max(worst, std::abs(X[e] - Y[e]));
  }
  return worst;
}

// Deterministic, self-contained, and portable across standard libraries, which
// <random>'s distributions are not.
struct Rng {
  std::uint64_t s = 0x9e3779b97f4a7c15ull;
  double next(double lo, double hi) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    const double u = static_cast<double>(s >> 11) / 9007199254740992.0;
    return lo + u * (hi - lo);
  }
};

// --- tests ------------------------------------------------------------------

// The core check: closed form versus measured, entry by entry, exactly.
void test_propagator_reproduces_the_stepper() {
  const Tableau T = cash_karp();
  // Dyadic, so t0 and h are themselves exact and nothing here depends on how a
  // decimal literal rounded. The stage times t0 + c_i h are not dyadic.
  const double t0 = 0.375, h = 0.25;

  for (int varying = 0; varying < 2; ++varying) {
    const LinearSystem sys = make_system(varying != 0);
    Mat Mm, Em, Mc, Ec;
    measure(sys, t0, h, Mm, Em);
    propagator(sys, T, t0, h, Mc, Ec);

    int wrong = 0;
    for (std::size_t e = 0; e < N * N; ++e) {
      if (Mc[e] != Mm[e]) { ++wrong; }
      if (Ec[e] != Em[e]) { ++wrong; }
    }
    const std::string which = varying ? "time-varying" : "autonomous";
    check(wrong == 0, which + ": propagator and stepper agree in all 18 entries");
    if (wrong != 0) {
      std::printf("       (%d entries differ; M off by %ld ulp, E by %ld ulp)\n",
                  wrong, worst_ulp(Mc, Mm), worst_ulp(Ec, Em));
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
          const std::size_t e = i * N + j;
          if (Mc[e] != Mm[e]) {
            std::printf("       M[%zu][%zu] closed-form %.17g measured %.17g\n",
                        i, j, Mc[e], Mm[e]);
          }
        }
      }
    }
  }

  // Not an accident of one (t0, h, A): sweep. Bitwise throughout, so this stays
  // an equality check rather than becoming a tolerance.
  Rng rng;
  long worstM = 0, worstE = 0;
  int trials = 0, disagreed = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    LinearSystem s;
    for (std::size_t i = 0; i < N * N; ++i) {
      s.P[i] = rng.next(-2.0, 2.0);
      s.Q[i] = (trial % 2) ? rng.next(-2.0, 2.0) : 0.0;
    }
    const double tt = rng.next(-2.0, 2.0), hh = rng.next(1e-4, 1.0);
    Mat Mm, Em, Mc, Ec;
    measure(s, tt, hh, Mm, Em);
    propagator(s, T, tt, hh, Mc, Ec);
    const long dm = worst_ulp(Mc, Mm), de = worst_ulp(Ec, Em);
    if (dm != 0 || de != 0) { ++disagreed; }
    worstM = std::max(worstM, dm);
    worstE = std::max(worstE, de);
    ++trials;
  }
  check(disagreed == 0,
        "and over " + std::to_string(trials) +
            " randomised (t0, h, A), still bitwise");
  if (disagreed != 0) {
    std::printf("       (%d of %d disagreed; worst M %ld ulp, worst E %ld ulp)\n",
                disagreed, trials, worstM, worstE);
  }
}

// The propagator describes what the *solver* does, not only what Step<> does.
void test_propagator_reproduces_the_solver() {
  const Tableau T = cash_karp();
  const double t0 = 0.375, h = 0.25;   // dyadic: (t0 + h) - t0 == h exactly
  const LinearSystem proto = make_system(true);

  Mat Ms{};
  for (std::size_t j = 0; j < N; ++j) {
    LinearSystem sys = proto;
    sys.t = t0;
    sys.y.assign(N, 0.0);
    sys.y[j] = 1.0;
    odelia::ode::Solver<LinearSystem> solver(sys, odelia::ode::OdeControl());
    solver.advance_fixed({t0, t0 + h});
    const std::vector<double> out = solver.state();
    for (std::size_t i = 0; i < N; ++i) { Ms[i * N + j] = out[i]; }
  }

  Mat Mc, Ec;
  propagator(proto, T, t0, h, Mc, Ec);
  check(worst_ulp(Mc, Ms) == 0,
        "a fixed step through Solver gives the same propagator");
}

// Which stages contribute to what. b_2 and b_5 are zero, so stages 2 and 5 do
// not enter y_{n+1} -- but stage 2 still feeds stages 3-6 through a_il, and
// stage 5 still enters the error estimate through b_5 - b*_5 = -277/14336. This
// measures each of those rather than asserting them.
void test_which_stages_contribute() {
  const Tableau T = cash_karp();
  const double t0 = 0.375, h = 0.25;
  const LinearSystem sys = make_system(true);

  Mat M0, E0;
  propagator(sys, T, t0, h, M0, E0);

  std::printf("       stage   b_i        b_i-b*_i    d|M| if b_i dropped   "
              "d|E| if (b-b*)_i dropped\n");
  for (std::size_t i = 0; i < 6; ++i) {
    Tableau Tb = T, Te = T;
    Tb.b[i] = 0.0;
    // Dropping the error weight without touching b: move bstar onto b.
    Te.bstar[i] = Te.b[i];
    Mat Mb, Eb, Me, Ee;
    propagator(sys, Tb, t0, h, Mb, Eb);
    propagator(sys, Te, t0, h, Me, Ee);
    std::printf("       %zu    %10.7f  %10.7f  %18.3e  %18.3e\n", i + 1, T.b[i],
                T.b[i] - T.bstar[i], max_abs_diff(Mb, M0), max_abs_diff(Ee, E0));
  }

  // Stages 2 and 5 carry no weight in the advanced solution.
  check(T.b[1] == 0.0 && T.b[4] == 0.0,
        "b_2 and b_5 are zero: stages 2 and 5 do not enter y_{n+1}");
  // Stage 5 nonetheless enters the error estimate.
  check(T.b[4] - T.bstar[4] != 0.0,
        "but stage 5 does enter the error estimate");
  check(T.b[1] - T.bstar[1] == 0.0,
        "and stage 2 enters neither");

  // "Contributes nothing" is only true of the two sums. Stage 2 feeds every
  // later stage; perturbing a_32 alone must move M.
  Tableau Ta = T;
  Ta.a[2][1] += 1e-3;
  Mat Ma, Ea;
  propagator(sys, Ta, t0, h, Ma, Ea);
  const double moved = max_abs_diff(Ma, M0);
  check(moved > 0.0,
        "yet stage 2 still reaches y_{n+1} through a_il (perturbing a_32 moves M)");
  std::printf("       (a_32 + 1e-3 moves M by %.3e)\n", moved);

  // Every abscissa is distinct and only the first is zero. Worth pinning: the
  // zero entries of this tableau are weights, and confusing the two is what
  // makes an autonomous test look adequate.
  bool distinct = true, only_first_zero = T.c[0] == 0.0;
  for (std::size_t i = 0; i < 6; ++i) {
    if (i > 0 && T.c[i] == 0.0) { only_first_zero = false; }
    for (std::size_t j = i + 1; j < 6; ++j) {
      if (T.c[i] == T.c[j]) { distinct = false; }
    }
  }
  check(distinct && only_first_zero,
        "the abscissae are distinct and only c_1 is zero");
}

// The justification for the whole design, measured. An abscissa is invisible on
// an autonomous system and visible on a time-varying one.
void test_abscissae_need_a_time_varying_system() {
  const Tableau T = cash_karp();
  const double t0 = 0.375, h = 0.25;
  const double delta = 0.1;

  for (int varying = 0; varying < 2; ++varying) {
    const LinearSystem sys = make_system(varying != 0);
    Mat Mm, Em;
    measure(sys, t0, h, Mm, Em);   // the real stepper, correct abscissae

    int invisible = 0;
    double smallest_visible = 0.0;
    std::printf("       %s system, c_i + %g:\n",
                varying ? "time-varying" : "autonomous", delta);
    for (std::size_t k = 1; k < 6; ++k) {
      Tableau Tp = T;
      Tp.c[k] += delta;
      Mat Mp, Ep;
      propagator(sys, Tp, t0, h, Mp, Ep);
      const double moved = max_abs_diff(Mp, Mm);
      if (moved == 0.0) {
        ++invisible;
      } else if (smallest_visible == 0.0 || moved < smallest_visible) {
        smallest_visible = moved;
      }
      std::printf("         c_%zu: max |M_wrong - M_stepper| = %.4g\n", k + 1,
                  moved);
    }

    if (varying) {
      check(invisible == 0,
            "on a time-varying system every abscissa is visible");
      // M is O(1) here, so a rounding-scale disagreement would be ~1e-16.
      std::printf("       (smallest visible discrepancy %.3e, i.e. %.0e times "
                  "the rounding scale of M)\n",
                  smallest_visible, smallest_visible / 1e-16);
    } else {
      check(invisible == 5,
            "on an autonomous system no abscissa is visible at all");
    }
  }
}

// A rejected attempt restores y and time, so it must cost nothing but work. The
// comparison is against a run that takes the same first step without rejecting.
void test_rejection_is_bit_neutral() {
  const double t_end = 2.0;
  auto control = [](double h0) {
    // tol_abs, tol_rel, a_y, a_dydt, h_min, h_max, h_init
    return odelia::ode::OdeControl(1e-8, 1e-8, 1.0, 0.0, 1e-12, 10.0, h0);
  };

  // h_init = 1.0 is far too large at this tolerance: the first step is rejected.
  derivs_calls = 0;
  odelia::ode::Solver<LinearSystem> rejecting(make_system(true), control(1.0));
  rejecting.advance_adaptive({0.0, t_end});
  const long calls_rejecting = derivs_calls;
  const std::vector<double> times_r = rejecting.times();
  const std::vector<double> state_r = rejecting.state();

  // Start where that run ended up after its retries. Beginning at t = 0 exactly
  // makes the first accepted step size readable as times[1].
  derivs_calls = 0;
  odelia::ode::Solver<LinearSystem> clean(make_system(true), control(times_r[1]));
  clean.advance_adaptive({0.0, t_end});
  const long calls_clean = derivs_calls;
  const std::vector<double> times_c = clean.times();
  const std::vector<double> state_c = clean.state();

  check(calls_rejecting > calls_clean,
        "the rejecting run really did reject (test is not vacuous)");
  std::printf("       (%ld vs %ld derivative evaluations over %zu accepted "
              "steps: %ld extra = %ld rejected attempts)\n",
              calls_rejecting, calls_clean, times_c.size() - 1,
              calls_rejecting - calls_clean, (calls_rejecting - calls_clean) / 6);

  bool same_times = times_r.size() == times_c.size();
  if (same_times) {
    for (std::size_t i = 0; i < times_r.size(); ++i) {
      if (times_r[i] != times_c[i]) { same_times = false; }
    }
  }
  check(same_times, "and recorded exactly the same times, bit for bit");

  bool same_state = state_r.size() == state_c.size();
  for (std::size_t i = 0; i < state_r.size() && same_state; ++i) {
    if (state_r[i] != state_c[i]) { same_state = false; }
  }
  check(same_state, "and landed on the same final state, bit for bit");
  if (!same_state) {
    for (std::size_t i = 0; i < state_r.size(); ++i) {
      std::printf("       y[%zu] rejecting %.17g clean %.17g (%ld ulp)\n", i,
                  state_r[i], state_c[i], ulp_dist(state_r[i], state_c[i]));
    }
  }
}

// Anything that differentiates this solver later has to replay the controller's
// step sizes rather than re-derive them, so: are they recoverable from a run?
// The solver records prev_times and nothing else, and differencing those does
// not return the step size that was used.
void test_step_size_sequence_is_recoverable() {
  odelia::ode::OdeControl control(1e-8, 1e-8, 1.0, 0.0, 1e-12, 10.0, 0.05);

  // step() with no time_max set never hits the final-step clamp, so every
  // interval here is a plain controller-chosen step.
  odelia::ode::Solver<LinearSystem> run(make_system(true), control);
  std::vector<std::vector<double>> Y{run.state()};
  while (run.time() < 2.0) {
    run.step();
    Y.push_back(run.state());
  }
  const std::vector<double> times = run.times();
  const std::size_t steps = times.size() - 1;

  // A run is reproducible: same inputs, same trajectory, bit for bit.
  odelia::ode::Solver<LinearSystem> again(make_system(true), control);
  std::vector<std::vector<double>> Y2{again.state()};
  while (again.time() < 2.0) {
    again.step();
    Y2.push_back(again.state());
  }
  bool identical = again.times() == times && Y2.size() == Y.size();
  for (std::size_t k = 0; k < Y.size() && identical; ++k) {
    for (std::size_t i = 0; i < N; ++i) {
      if (Y[k][i] != Y2[k][i]) { identical = false; }
    }
  }
  check(identical, "an adaptive run is reproducible bit for bit");

  // Each step, replayed from its own starting state at dt = t_{k+1} - t_k.
  std::size_t exact = 0;
  double worst = 0.0;
  long worst_offset = 0, bound = 0;
  bool all_found = true, within_bound = true;
  for (std::size_t k = 0; k < steps; ++k) {
    const double t0 = times[k], dt = times[k + 1] - times[k];

    // How much of dt a recorded time can hold. t_{k+1} = t_k + dt is rounded to
    // ulp(t_k), so differencing recovers dt only to ulp(t_k) -- which is
    // ulp(t_k)/ulp(dt) representable steps of dt. That ratio is what the search
    // below has to come in under, per step; it is derived, not chosen.
    const double inf = std::numeric_limits<double>::infinity();
    long bound_k = 1;
    if (t0 != 0.0) {
      const double ulp_t = std::nextafter(t0, inf) - t0;
      const double ulp_dt = std::nextafter(dt, inf) - dt;
      bound_k = static_cast<long>(ulp_t / ulp_dt);
    }
    bound = std::max(bound, bound_k);

    // Search neighbouring representable dt for one that reproduces the step
    // exactly. Finding one says the stepper is deterministic and replayable and
    // that dt is the only thing missing; the offset says how much of dt the
    // recorded times threw away.
    long found = 0;
    bool ok_here = false;
    for (long off = 0; off <= 64 && !ok_here; ++off) {
      for (int sign = 0; sign < 2 && !ok_here; ++sign) {
        if (off == 0 && sign == 1) { continue; }
        double d = dt;
        for (long m = 0; m < off; ++m) {
          d = std::nextafter(d, sign ? -inf : inf);
        }
        LinearSystem sys = make_system(true);
        std::vector<double> y = Y[k], dydt_in(N), yerr(N), dydt_out(N);
        odelia::ode::derivs(sys, y, dydt_in, t0);
        odelia::ode::Step<LinearSystem> stepper;
        stepper.resize(N);
        stepper.step(sys, t0, d, y, yerr, dydt_in, dydt_out);
        bool same = true;
        for (std::size_t i = 0; i < N; ++i) {
          if (y[i] != Y[k + 1][i]) { same = false; }
        }
        if (same) {
          ok_here = true;
          found = sign ? -off : off;
        }
        if (off == 0) {
          if (same) { ++exact; }
          for (std::size_t i = 0; i < N; ++i) {
            worst = std::max(worst, std::abs(y[i] - Y[k + 1][i]));
          }
        }
      }
    }
    if (!ok_here) { all_found = false; }
    if (std::labs(found) > bound_k) { within_bound = false; }
    worst_offset = std::max(worst_offset, std::labs(found));
  }

  check(all_found,
        "every recorded step is reproduced exactly by one Step at some dt");
  check(within_bound,
        "and the dt it needs is within the ulp(t)/ulp(dt) the times can hold");
  std::printf("       (%zu of %zu steps reproduced at dt = t_{k+1} - t_k; the "
              "rest need dt up to %ld ulp away, bound %ld; worst state "
              "deviation at the differenced dt %.3g)\n",
              exact, steps, worst_offset, bound, worst);

  // Replaying the recorded times through advance_fixed -- the only replay the
  // API offers -- inherits that loss.
  odelia::ode::Solver<LinearSystem> replay(make_system(true), control);
  replay.advance_fixed(times);
  const std::vector<double> state_replay = replay.state();
  long replay_ulp = 0;
  for (std::size_t i = 0; i < N; ++i) {
    replay_ulp = std::max(replay_ulp, ulp_dist(Y.back()[i], state_replay[i]));
  }
  // Reported, not asserted: a test should not pin a shortfall in place. The
  // assertions above are the durable ones -- each step is exactly replayable,
  // and dt is the only thing the recorded output does not carry.
  std::printf("       (advance_fixed on the recorded times lands %ld ulp from "
              "the adaptive run; reproducing it exactly needs the step sizes "
              "recorded, not the times)\n", replay_ulp);
}

} // namespace

int main() {
  std::printf("odelia RKCK stepper, closed-form propagator exactness\n");
  test_propagator_reproduces_the_stepper();
  test_propagator_reproduces_the_solver();
  test_which_stages_contribute();
  test_abscissae_need_a_time_varying_system();
  test_rejection_is_bit_neutral();
  test_step_size_sequence_is_recoverable();
  if (failures == 0) {
    std::printf("all checks passed\n");
    return 0;
  }
  std::printf("%d failure(s)\n", failures);
  return 1;
}
