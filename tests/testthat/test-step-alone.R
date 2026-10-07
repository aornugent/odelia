# Tests for a step that takes a System's block alone: where a stiff block reads
# the rest only through a few inputs, the rest takes long steps while the block
# takes inner steps of its own, which replays and the sweep take as recorded.

# A population x drawing on a two-layer store w, which a seasonal source fills
# and drainage K w^2 carries layer to layer: stiff in w where K is large. The
# uptake c x w is what the rest of the state hands the store. With `Alone` false
# it is the same System naming no block.
store_system <- '
  static std::vector<std::vector<double>> last_dense;

  template <typename T, bool Alone = true>
  class Store {
  public:
    using value_type = T;
    Store(T a_ = T(0), T m_ = T(0), T K_ = T(0), T s0_ = T(0), T c_ = T(0))
      : a(a_), m(m_), K(K_), s0(s0_), c(c_) { reset(); }

    template <typename, bool> friend class Store;

    template <class S2>
    void assign_from(const Store<S2, Alone>& src) {
      a = T(odelia::util::to_passive(src.a));
      m = T(odelia::util::to_passive(src.m));
      K = T(odelia::util::to_passive(src.K));
      s0 = T(odelia::util::to_passive(src.s0));
      c = T(odelia::util::to_passive(src.c));
      x = T(odelia::util::to_passive(src.x));
      w1 = T(odelia::util::to_passive(src.w1));
      w2 = T(odelia::util::to_passive(src.w2));
      time = src.time;
      alone_share = src.alone_share;
      fail_after = src.fail_after;
      compute_rates();
    }

    template <class S2>
    Store<S2, Alone> rebind_from() const {
      Store<S2, Alone> out;
      out.assign_from(*this);
      return out;
    }

    std::vector<T*> ad_parameters() { return {&a, &m, &K, &s0, &c}; }

    template <class F>
    void for_each_active(F&& f) {
      f(a); f(m); f(K); f(s0); f(c);
      f(x); f(w1); f(w2);
      f(dx); f(dw1); f(dw2); f(u1); f(u2);
    }

    size_t ode_size() const { return 3; }
    double ode_time() const { return time; }
    void reset() { x = 1.0; w1 = 0.5; w2 = 0.5; time = 0.0; compute_rates(); }

    template <typename It> It set_ode_state(It it, double time_) {
      time = time_;
      x = *it++;
      w1 = *it++;
      w2 = *it++;
      compute_rates();
      return it;
    }
    template <typename It> It ode_state(It it) const {
      *it++ = x; *it++ = w1; *it++ = w2;
      return it;
    }
    template <typename It> It ode_rates(It it) const {
      *it++ = dx; *it++ = dw1; *it++ = dw2;
      return it;
    }

    // The store, from index 1, taken alone where the uptake is under
    // alone_share of the water moving through it.
    std::pair<std::size_t, std::size_t> alone_block() const requires Alone {
      return {1, 2};
    }
    bool steps_alone() const requires Alone {
      const double in1 = odelia::util::to_passive(inflow_at(time));
      const double k = odelia::util::to_passive(K);
      const double p1 = odelia::util::to_passive(w1), p2 = odelia::util::to_passive(w2);
      const double v1 = std::abs(odelia::util::to_passive(u1));
      const double v2 = std::abs(odelia::util::to_passive(u2));
      const double moving = in1 + 2.0 * k * p1 * p1 + k * p2 * p2 + v1 + v2;
      return (v1 + v2) / moving < alone_share;
    }
    void alone_inputs(std::vector<T>& u) const requires Alone { u = {u1, u2}; }
    template <class U>
    void alone_rates(double t, const std::vector<U>& w, const std::vector<U>& u,
                     std::vector<U>& out) const requires Alone {
      store_rates(t, w, out);
      out[0] -= u[0];
      out[1] -= u[1];
      if (t > fail_after) {
        out[0] = U(std::nan(""));
      }
    }

    // Splits nothing, and keeps what the dense output gives for the store at the
    // sample fractions of the step just taken.
    void sign_values(std::vector<double>& out) const { out.clear(); }
    template <class Step, class Samples, class Record>
    bool split_sign_changes(const Step& step, Samples&, Record&) {
      last_dense.clear();
      for (double u : step.sample_fractions) {
        std::vector<T> b(2);
        step.dense_state(u, 1, b);
        last_dense.push_back({odelia::util::to_passive(b[0]),
                              odelia::util::to_passive(b[1])});
      }
      return false;
    }
    template <class Step, class Row, class Samples, class Record>
    void take_recorded_splits(const Step&, const Row&, Samples&, Record&) {}
    template <class Step, class Samples, class Record>
    void split_as_recorded(const Step&, const Samples&, const Record&) {}

    double alone_share = 0.0;
    // Past this time the rates of the store alone are NaN.
    double fail_after = 1e300;

  private:
    template <class U> U at(const T& v) const {
      if constexpr (std::is_same_v<U, T>) {
        return v;
      } else {
        return odelia::util::to_passive(v);
      }
    }
    T inflow_at(double t) const { return s0 * (1.0 + 0.5 * std::sin(2.0 * M_PI * t)); }
    template <class U>
    void store_rates(double t, const std::vector<U>& w, std::vector<U>& out) const {
      const U k = at<U>(K);
      const U inflow = at<U>(s0) * (1.0 + 0.5 * std::sin(2.0 * M_PI * t));
      out.resize(2);
      out[0] = inflow - k * w[0] * w[0];
      out[1] = k * w[0] * w[0] - k * w[1] * w[1];
    }
    void compute_rates() {
      std::vector<T> w{w1, w2}, f(2);
      store_rates(time, w, f);
      u1 = c * x * w1;
      u2 = c * x * w2;
      dx = a * x * (w1 + w2) - m * x;
      dw1 = f[0] - u1;
      dw2 = f[1] - u2;
    }

    T a, m, K, s0, c;
    T x = 0, w1 = 0, w2 = 0, dx = 0, dw1 = 0, dw2 = 0, u1 = 0, u2 = 0;
    double time = 0;
  };
'

compile_alone_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = paste0('
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <algorithm>
    #include <cmath>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <odelia/tangent.hpp>
    ', store_system, '

    using namespace odelia;
    using StoreA = Store<double, true>;
    using StoreP = Store<double, false>;

    static_assert(ode::StepsBlockAlone<StoreA>);
    static_assert(!ode::StepsBlockAlone<StoreP>);
    static_assert(ode::SplitsSignChanges<StoreA>);

    ode::OdeControl control_at(double tol) {
      ode::OdeControl ctl;
      ctl.set_tol_abs(tol);
      ctl.set_tol_rel(tol);
      return ctl;
    }

    StoreA store(const std::vector<double>& p, double share) {
      StoreA s(p[0], p[1], p[2], p[3], p[4]);
      s.alone_share = share;
      return s;
    }

    std::vector<ode::instruction> program_of(const std::vector<double>& pars,
                                             const std::vector<double>& y0,
                                             double t_end, double tol,
                                             double share) {
      ode::Solver<StoreA> solver(store(pars, share), control_at(tol));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      return solver.schedule();
    }

    // [[Rcpp::export]]
    Rcpp::List store_run(double share, double tol, double t_end,
                         std::vector<double> pars, std::vector<double> y0,
                         double fail_after = 1e300) {
      const std::vector<double> stops{1.0 / 5.0, 0.25, 0.3, 0.5, 3.0 / 5.0,
                                      0.75, 7.0 / 8.0, 1.0};
      StoreA s = store(pars, share);
      s.fail_after = fail_after;
      ode::Solver<StoreA> solver(s, control_at(tol));
      solver.set_keep_states(true);
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      const std::vector<ode::instruction> program = solver.schedule();
      int alone = 0, inner = 0;
      bool stops_landed = true, ascending = true, sized = true;
      for (const ode::instruction& row : program) {
        if (row.alone.slope.empty()) {
          continue;
        }
        ++alone;
        inner += row.alone.ends.size();
        sized = sized && row.alone.slope.size() == 2;
        ascending = ascending && std::is_sorted(row.alone.ends.begin(),
                                                row.alone.ends.end());
        for (double stop : stops) {
          stops_landed = stops_landed &&
            std::find(row.alone.ends.begin(), row.alone.ends.end(), stop) !=
              row.alone.ends.end();
        }
      }
      // The last step: where it started, its size, its slope, and what the dense
      // output gave for the store at the sample fractions.
      const auto rec = solver.recording();
      const auto& last = rec[rec.size() - 1];
      return Rcpp::List::create(
        Rcpp::_["y"] = solver.state(),
        Rcpp::_["steps"] = (int) program.size() - 1,
        Rcpp::_["alone"] = alone, Rcpp::_["inner"] = inner,
        Rcpp::_["stops_landed"] = stops_landed, Rcpp::_["ascending"] = ascending,
        Rcpp::_["sized"] = sized,
        Rcpp::_["last_start"] = rec[rec.size() - 2].state,
        Rcpp::_["last_time"] = rec[rec.size() - 2].time,
        Rcpp::_["last_size"] = last.step_size,
        Rcpp::_["last_slope"] = last.alone.slope,
        Rcpp::_["last_dense"] = last_dense);
    }

    // [[Rcpp::export]]
    Rcpp::List store_plain_run(double tol, double t_end,
                               std::vector<double> pars,
                               std::vector<double> y0) {
      ode::Solver<StoreP> solver(StoreP(pars[0], pars[1], pars[2], pars[3],
                                        pars[4]), control_at(tol));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      return Rcpp::List::create(Rcpp::_["y"] = solver.state(),
                                Rcpp::_["steps"] = (int) solver.times().size() - 1);
    }

    // The program a run at `program_pars` took, replayed at `pars` from `y0`:
    // where it lands, and whether its own rows took the inner steps the run took.
    // [[Rcpp::export]]
    Rcpp::List store_replay(std::vector<double> program_pars,
                            std::vector<double> pars, std::vector<double> y0,
                            std::vector<double> y0_program, double t_end,
                            double tol, double share, std::string method) {
      const std::vector<ode::instruction> program =
        program_of(program_pars, y0_program, t_end, tol, share);
      ode::Solver<StoreA> replay(store(pars, share), control_at(tol),
                                 method == "rodas" ? ode::Method::rodas
                                                   : ode::Method::rkck);
      replay.set_state(y0, 0.0);
      replay.advance_recorded(program);
      const std::vector<ode::instruction> taken = replay.schedule();
      bool same_steps = taken.size() == program.size();
      for (size_t k = 0; same_steps && k < taken.size(); ++k) {
        same_steps = taken[k].alone.ends == program[k].alone.ends &&
                     taken[k].alone.slope == program[k].alone.slope;
      }
      return Rcpp::List::create(Rcpp::_["y"] = replay.state(),
                                Rcpp::_["same_steps"] = same_steps);
    }

    // A walk at the tangent scalar over a run that took the store alone, which it
    // refuses.
    // [[Rcpp::export]]
    void store_tangent_walk(std::vector<double> pars, std::vector<double> y0,
                            double t_end, double tol, double share) {
      ode::Solver<StoreA> run(store(pars, share), control_at(tol));
      run.set_keep_states(true);
      run.set_state(y0, 0.0);
      run.advance_adaptive({0.0, t_end});
      using tangent = ode::tangent_scalar<>;
      ode::Solver<Store<tangent, true>> walk(
        store(pars, share).rebind_from<tangent>(), control_at(tol));
      walk.set_collect(false);
      walk.advance_recorded(run.recording());
    }

    // [[Rcpp::export]]
    Rcpp::List store_adjoint(std::vector<double> pars, std::vector<double> y0,
                             double t_end, std::vector<double> lambda_end,
                             double tol, double share) {
      ode::Solver<StoreA> run(store(pars, share), control_at(tol));
      run.set_keep_states(true);
      run.set_state(y0, 0.0);
      run.advance_adaptive({0.0, t_end});
      ode::adjoint_rows lambda = ode::adjoint_rows::one_row(lambda_end);
      ode::adjoint_rows rows(1, pars.size());
      run.clear_recorded_rates();
      run.solve_adjoint(lambda, rows);
      return Rcpp::List::create(
        Rcpp::_["lambda"] = lambda.to_rows()[0],
        Rcpp::_["parameter_adjoint"] = rows.to_rows()[0],
        Rcpp::_["n_steps"] = (int) run.recording().size() - 1,
        Rcpp::_["recorded_rates"] = (int) run.recorded_rates());
    }
  '), verbose = FALSE)
}

store_pars <- c(1.0, 0.8, 1e4, 200, 0.3)
store_y <- c(1.0, 0.5, 0.5)

testthat::test_that("with the share at zero, every step is the plain System's", {
  compile_alone_interface()
  off <- store_run(0.0, 1e-6, 0.5, store_pars, store_y)
  plain <- store_plain_run(1e-6, 0.5, store_pars, store_y)
  expect_identical(off$alone, 0L)
  expect_identical(off$steps, plain$steps)
  expect_identical(off$y, plain$y)
})

testthat::test_that("taken alone, the stiff store lets the population take long steps, as accurately", {
  compile_alone_interface()
  reference <- store_plain_run(1e-12, 2.0, store_pars, store_y)$y
  ck <- store_plain_run(1e-6, 2.0, store_pars, store_y)
  alone <- store_run(0.1, 1e-6, 2.0, store_pars, store_y)
  # Cash-Karp is held to its stability boundary by the store's drainage; every
  # step taken alone takes inner steps of its own instead.
  expect_gt(ck$steps, 1000L)
  expect_identical(alone$alone, alone$steps)
  expect_lt(alone$steps, ck$steps / 10)
  expect_lt(max(abs(alone$y - reference)), 1e-5)
  # It converges on the same answer as its tolerance tightens.
  tight <- store_run(0.1, 1e-9, 2.0, store_pars, store_y)
  expect_lt(max(abs(tight$y - reference)), 1e-8)
})

testthat::test_that("each row records its slope and inner steps, landing on every stop", {
  compile_alone_interface()
  r <- store_run(0.1, 1e-6, 2.0, store_pars, store_y)
  expect_true(r$sized)
  expect_true(r$ascending)
  expect_true(r$stops_landed)
  expect_gt(r$inner, 8L * r$alone)
})

testthat::test_that("the dense output reads the store from the predictor's inner steps", {
  compile_alone_interface()
  r <- store_run(0.1, 1e-6, 2.0, store_pars, store_y)
  # The predictor over the last step, independently: the store alone under the
  # uptake extrapolated by the recorded slope, by classical Runge-Kutta.
  y0 <- r$last_start
  t0 <- r$last_time
  h <- r$last_size
  u0 <- store_pars[5] * y0[1] * y0[2:3]
  u1 <- pmax(0, u0 + h * r$last_slope)
  f <- function(t, w) {
    inflow <- store_pars[4] * (1 + 0.5 * sin(2 * pi * t))
    k <- store_pars[3]
    c(inflow - k * w[1]^2, k * w[1]^2 - k * w[2]^2) - (u0 + (u1 - u0) * (t - t0) / h)
  }
  w <- y0[2:3]
  n <- 8000
  dt <- h / n
  expected <- list(w)
  for (i in seq_len(n) - 1) {
    t <- t0 + i * dt
    k1 <- f(t, w)
    k2 <- f(t + dt / 2, w + dt / 2 * k1)
    k3 <- f(t + dt / 2, w + dt / 2 * k2)
    k4 <- f(t + dt, w + dt * k3)
    w <- w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if ((i + 1) %% (n / 4) == 0) expected[[length(expected) + 1]] <- w
  }
  for (m in seq_along(expected)) {
    expect_equal(r$last_dense[[m]], expected[[m]], tolerance = 1e-7)
  }
})

testthat::test_that("a store whose rates fail at every inner step stops the run, saying why", {
  compile_alone_interface()
  expect_error(store_run(0.1, 1e-6, 2.0, store_pars, store_y, fail_after = 1.0),
               "block alone is not finite at the smallest step")
})

testthat::test_that("a replay takes the run's inner steps, at the run's parameters to the bit", {
  compile_alone_interface()
  run <- store_run(0.1, 1e-6, 2.0, store_pars, store_y)
  same <- store_replay(store_pars, store_pars, store_y, store_y, 2.0, 1e-6, 0.1, "rkck")
  expect_identical(same$y, run$y)
  expect_true(same$same_steps)
  moved <- store_pars * c(1, 1, 1.01, 1, 1)
  away <- store_replay(store_pars, moved, store_y, store_y, 2.0, 1e-6, 0.1, "rkck")
  expect_true(away$same_steps)
  expect_false(identical(away$y, run$y))
  expect_error(store_replay(store_pars, store_pars, store_y, store_y, 2.0, 1e-6,
                            0.1, "rodas"),
               "cannot take a System's block alone")
})

testthat::test_that("a walk at another scalar refuses a step that took the store alone", {
  compile_alone_interface()
  expect_error(store_tangent_walk(store_pars, store_y, 0.5, 1e-6, 0.1),
               "cannot take a recorded step that took the System's block alone")
})

testthat::test_that("the sweep through steps taken alone matches central differences of the replayed program", {
  compile_alone_interface()
  t_end <- 2.0
  tol <- 1e-6
  lambda_end <- c(0.4, -1.3, 0.7)
  r <- store_adjoint(store_pars, store_y, t_end, lambda_end, tol, 0.1)
  expect_identical(r$recorded_rates, 6L * r$n_steps)

  eps <- 1e-4
  replay <- function(pars, y0) {
    store_replay(store_pars, pars, y0, store_y, t_end, tol, 0.1, "rkck")$y
  }
  expected_state <- sapply(seq_along(store_y), function(j) {
    e <- eps * store_y[j]
    up <- store_y; up[j] <- up[j] + e
    down <- store_y; down[j] <- down[j] - e
    sum((replay(store_pars, up) - replay(store_pars, down)) / (2 * e) * lambda_end)
  })
  expected_pars <- sapply(seq_along(store_pars), function(j) {
    e <- eps * store_pars[j]
    up <- store_pars; up[j] <- up[j] + e
    down <- store_pars; down[j] <- down[j] - e
    sum((replay(up, store_y) - replay(down, store_y)) / (2 * e) * lambda_end)
  })
  expect_equal(r$lambda / expected_state, rep(1, 3), tolerance = 1e-5)
  expect_equal(r$parameter_adjoint / expected_pars, rep(1, 5), tolerance = 1e-5)
})
