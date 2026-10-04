# Tests that the stepper's dense output is fourth order and ends on the step's
# solution, that a part is integrated in pieces between its sign changes, once
# where its sign value changes sign across a step and twice where it dips inside
# one, a stage holding the other sign or not, that the record says what each part
# did, and that a System naming no parts runs bit for bit as one without them.

compile_split_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <algorithm>
    #include <cmath>
    #include <vector>
    #include <odelia/ode_solver.hpp>

    using namespace odelia;

    // y1 = cos t, y2 = -sin t.
    struct Oscillator {
      using value_type = double;
      size_t ode_size() const { return 2; }
      double ode_time() const { return time; }
      template <typename It> It set_ode_state(It it, double t) {
        for (double& v : y) v = *it++;
        time = t;
        return it;
      }
      template <typename It> It ode_state(It it) const {
        for (double v : y) *it++ = v;
        return it;
      }
      template <typename It> It ode_rates(It it) {
        *it++ = y[1];
        *it++ = -y[0];
        return it;
      }
      std::vector<double> y{1.0, 0.0};
      double time = 0.0;
    };

    // One step of size h from y = (1, 0), then the dense output at each u, less
    // the solution there; the last row is the end less the dense output at 1.
    // [[Rcpp::export]]
    Rcpp::NumericMatrix oscillator_dense(double h, std::vector<double> u) {
      Oscillator sys;
      ode::Step<Oscillator> stepper;
      stepper.resize(2);
      ode::Step<Oscillator>::solved_row solved;
      std::vector<double> y{1.0, 0.0}, yerr(2), dydt_in(2), dydt_out(2), out(2);
      ode::derivs(sys, y, dydt_in, 0.0);
      stepper.step(sys, solved, 0.0, h, y, yerr, dydt_in, dydt_out);
      Rcpp::NumericMatrix ret(static_cast<int>(u.size()) + 1, 2);
      for (size_t i = 0; i < u.size(); ++i) {
        stepper.dense_state(u[i], h, dydt_out, out);
        ret(i, 0) = out[0] - std::cos(u[i] * h);
        ret(i, 1) = out[1] + std::sin(u[i] * h);
      }
      stepper.dense_state(1.0, h, dydt_out, out);
      ret(u.size(), 0) = y[0] - out[0];
      ret(u.size(), 1) = y[1] - out[1];
      return ret;
    }

    // Four parts of one component, x_p rising at max(g_p, 0), then z = t and
    // w = t^2 + 2t, which no part holds. g_0 = z - 0.3; g_1 is below zero on
    // (0.55, 0.65), where the stage at 0.6 falls, and g_2 on (0.06, 0.14), before
    // the first stage. g_3 is below zero for w in (0.3, 0.42), t in (0.140, 0.192),
    // so at w = 0.4 in the first stage, but not at w(0.2) = 0.44 on the dense
    // output. Only Kinked<true> names its parts.
    template <bool Parts>
    struct Kinked {
      using value_type = double;
      static double gate(std::size_t p, double z, double w) {
        return p == 0   ? z - 0.3
               : p == 1 ? (z - 0.6) * (z - 0.6) - 0.0025
               : p == 2 ? (z - 0.1) * (z - 0.1) - 0.0016
                        : (w - 0.3) * (w - 0.42);
      }
      size_t ode_size() const { return 6; }
      double ode_time() const { return time; }
      void reset() {
        y.assign(6, 0.0);
        time = 0.0;
      }
      template <typename It> It set_ode_state(It it, double t) {
        for (double& v : y) v = *it++;
        time = t;
        return it;
      }
      template <typename It> It ode_state(It it) const {
        for (double v : y) *it++ = v;
        return it;
      }
      template <typename It> It ode_rates(It it) {
        for (std::size_t p = 0; p < 4; ++p) {
          g[p] = gate(p, y[4], y[5]);
          *it++ = std::max(g[p], 0.0);
        }
        *it++ = 1.0;
        *it++ = 2.0 * (y[4] + 1.0);
        return it;
      }
      void sign_values(std::vector<double>& out) const requires Parts {
        if (splits) {
          out.assign(g.begin(), g.end());
        } else {
          out.clear();
        }
      }
      std::size_t part_width() const requires Parts { return 1; }
      double part_rates(std::size_t p, const std::vector<double>& state, double,
                        std::vector<double>& out) requires Parts {
        const double v = gate(p, state[4], state[5]);
        out[0] = std::max(v, 0.0);
        return v;
      }

      bool splits = true;
      std::vector<double> y{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      std::vector<double> g{0.0, 0.0, 0.0, 0.0};
      double time = 0.0;
    };

    static Rcpp::List record(const ode::split_record& r) {
      return Rcpp::List::create(
        Rcpp::Named("total") = static_cast<int>(r.total()),
        Rcpp::Named("split") = std::vector<int>(r.split.begin(), r.split.end()),
        Rcpp::Named("searched") =
          std::vector<int>(r.searched.begin(), r.searched.end()),
        Rcpp::Named("least_rate") = r.least_rate,
        Rcpp::Named("least_rate_time") = r.least_rate_time,
        Rcpp::Named("least_rate_part") = static_cast<int>(r.least_rate_part));
    }

    template <bool Parts>
    static ode::Solver<Kinked<Parts>> kinked_solver(bool splits, bool keep) {
      Kinked<Parts> sys;
      sys.splits = splits;
      ode::Solver<Kinked<Parts>> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      solver.set_keep_states(keep);
      solver.reset();
      return solver;
    }

    // One step pinned over [0, 1].
    // [[Rcpp::export]]
    Rcpp::List kinked_one_step(bool splits) {
      auto solver = kinked_solver<true>(splits, false);
      solver.advance_fixed({0.0, 1.0});
      return Rcpp::List::create(Rcpp::Named("state") = solver.state(),
                                Rcpp::Named("splits") = record(solver.splits()));
    }

    // An adaptive run over [0, 2]: with the parts named, unnamed, or with no
    // parts at all; then a walk over the recording of the first.
    // [[Rcpp::export]]
    Rcpp::List kinked_runs() {
      auto split = kinked_solver<true>(true, true);
      split.advance_adaptive({0.0, 2.0});
      auto unnamed = kinked_solver<true>(false, false);
      unnamed.advance_adaptive({0.0, 2.0});
      auto none = kinked_solver<false>(false, false);
      none.advance_adaptive({0.0, 2.0});

      std::vector<ode::step_record<Kinked<true>>> rows(
        split.recording().begin(), split.recording().end());
      auto walk = kinked_solver<true>(true, true);
      walk.advance_recorded(rows);
      std::vector<double> run_states, walk_states;
      for (const auto& row : split.recording()) {
        run_states.insert(run_states.end(), row.state.begin(), row.state.end());
      }
      for (const auto& row : walk.recording()) {
        walk_states.insert(walk_states.end(), row.state.begin(), row.state.end());
      }

      auto run = [](auto& s) {
        return Rcpp::List::create(
          Rcpp::Named("times") = s.times(),
          Rcpp::Named("step_sizes") = s.step_sizes(),
          Rcpp::Named("state") = s.state(),
          Rcpp::Named("splits") = record(s.splits()));
      };
      return Rcpp::List::create(Rcpp::Named("split") = run(split),
                                Rcpp::Named("unnamed") = run(unnamed),
                                Rcpp::Named("none") = run(none),
                                Rcpp::Named("walk") = run(walk),
                                Rcpp::Named("run_states") = run_states,
                                Rcpp::Named("walk_states") = walk_states);
    }
  ', verbose = FALSE)
}

bits <- function(x) vapply(x, function(v) paste(writeBin(v, raw()), collapse = ""),
                           character(1))

test_that("the dense output is fourth order and ends on the step's solution", {
  compile_split_interface()
  u <- c(0.25, 0.5, 0.75)
  coarse <- oscillator_dense(0.2, u)
  fine <- oscillator_dense(0.1, u)
  # A fourth-order output errs by h^5 inside the step, so at least 32 times less
  # at h / 2; the cosine's leading term vanishes, and it falls 64 times.
  ratio <- abs(coarse[1:3, ]) / abs(fine[1:3, ])
  expect_true(all(ratio > 24))
  expect_true(all(ratio[, 2] < 40))
  expect_true(all(abs(coarse[4, ]) < 1e-15))
  expect_true(all(abs(fine[4, ]) < 1e-15))
})

test_that("one step is integrated in pieces at a sign change and at a dip", {
  compile_split_interface()
  # x_0 = (1 - 0.3)^2 / 2; x_1 is g_1's integral less its part inside the dip,
  # 0.28 / 3 - 0.0025 + 4 * 0.05^3 / 3, and x_2 likewise, 0.730256 / 3 - 0.0016.
  # With s = 1 + t, g_3 = (s^2 - 1.3)(s^2 - 1.42), whose integral is F. Each piece
  # is a polynomial the stepper integrates exactly.
  F <- function(s) s^5 / 5 - 2.72 * s^3 / 3 + 1.846 * s
  x3 <- F(2) - F(1) - (F(sqrt(1.42)) - F(sqrt(1.3)))
  exact <- c(0.245, 0.091, 0.730256 / 3 - 0.0016, x3, 1, 3)
  split <- kinked_one_step(TRUE)
  plain <- kinked_one_step(FALSE)
  expect_equal(split$state, exact, tolerance = 1e-13)
  expect_equal(split$splits$total, 4L)
  expect_gt(abs(plain$state[1] - exact[1]), 1e-2)
  expect_gt(abs(plain$state[2] - exact[2]), 1e-4)
  expect_gt(abs(plain$state[3] - exact[3]), 1e-5)
  expect_gt(abs(plain$state[4] - exact[4]), 1e-6)
  expect_equal(plain$splits$total, 0L)
})

test_that("a dip no stage holds is found beside the reading nearest zero", {
  compile_split_interface()
  # g_2 reads 0.0084 at the step's start and at the first stage, within 2% of its
  # readings' spread of 0.8, and is below zero only between them. g_3's first
  # stage reads -0.002, which the dense output does not hold, so its gaps are
  # searched. g_1's dip holds the stage at 0.6, so it is cut without a search.
  r <- kinked_one_step(TRUE)$splits
  expect_equal(r$split, c(1L, 1L, 1L, 1L))
  expect_equal(r$searched, c(0L, 0L, 1L, 1L))
  # The slowest crossings are g_2's, at 0.06 and 0.14 with |dg_2/dz| = 0.08, read
  # from a difference over at least 1e-6 of the step, so to about 1e-4.
  expect_equal(r$least_rate, 0.08, tolerance = 1e-3)
  expect_lt(min(abs(r$least_rate_time - c(0.06, 0.14))), 1e-9)
  expect_equal(r$least_rate_part, 2L)
})

test_that("a System naming no parts runs bit for bit as one without them", {
  compile_split_interface()
  res <- kinked_runs()
  expect_identical(bits(res$unnamed$times), bits(res$none$times))
  expect_identical(bits(res$unnamed$step_sizes), bits(res$none$step_sizes))
  expect_identical(bits(res$unnamed$state), bits(res$none$state))
  # Splitting moves the run: x_0 reaches (2 - 0.3)^2 / 2 exactly.
  expect_equal(res$split$state[1], 1.445, tolerance = 1e-13)
  expect_false(identical(res$split$state, res$none$state))
  expect_gt(res$split$splits$total, 0L)
  expect_equal(res$unnamed$splits$total, 0L)
})

test_that("a walk over a split run's recording splits again, bit for bit", {
  compile_split_interface()
  res <- kinked_runs()
  expect_identical(bits(res$walk_states), bits(res$run_states))
  # Only the steps a run keeps are split, so the walk records what the run did.
  expect_identical(res$walk$splits, res$split$splits)
})
