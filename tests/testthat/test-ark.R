# Tests for the additive ARK4(3)6L[2]SA stepper (method = ark): implicit in the
# stiff block a System names and explicit elsewhere. It steps a stiff block in a
# fraction of Cash-Karp's steps at the same accuracy; its sweep matches central
# differences of the replayed run; and the block's error estimate is the block
# alone's.

# A population x drawing on a two-layer store w, which a seasonal source fills
# and drainage K w^2 carries layer to layer: stiff in w where K is large. The
# store's uptake c x w is what the rest of the state hands the block.
store_system <- '
  static int store_alone_calls = 0;

  template <typename T>
  class Store {
  public:
    using value_type = T;
    Store(T a_ = T(0), T m_ = T(0), T K_ = T(0), T s0_ = T(0), T c_ = T(0))
      : a(a_), m(m_), K(K_), s0(s0_), c(c_) { reset(); }

    template <typename> friend class Store;

    template <class S2>
    void assign_from(const Store<S2>& src) {
      a = T(odelia::util::to_passive(src.a));
      m = T(odelia::util::to_passive(src.m));
      K = T(odelia::util::to_passive(src.K));
      s0 = T(odelia::util::to_passive(src.s0));
      c = T(odelia::util::to_passive(src.c));
      x = T(odelia::util::to_passive(src.x));
      w1 = T(odelia::util::to_passive(src.w1));
      w2 = T(odelia::util::to_passive(src.w2));
      time = src.time;
      compute_rates();
    }

    template <class S2>
    Store<S2> rebind_from() const {
      Store<S2> out;
      out.assign_from(*this);
      return out;
    }

    std::vector<T*> ad_parameters() { return {&a, &m, &K, &s0, &c}; }

    template <class F>
    void for_each_active(F&& f) {
      f(a); f(m); f(K); f(s0); f(c);
      f(x); f(w1); f(w2);
      f(dx); f(dw1); f(dw2);
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

    // The store: the state from index 1, two layers.
    std::pair<std::size_t, std::size_t> stiff_block() const { return {1, 2}; }
    template <class U>
    void stiff_rates(double t, const std::vector<U>& w, std::vector<U>& out) const {
      const U k = at<U>(K);
      const U inflow = at<U>(s0) * (1.0 + 0.5 * std::sin(2.0 * M_PI * t));
      out.resize(2);
      out[0] = inflow - k * w[0] * w[0];
      out[1] = k * w[0] * w[0] - k * w[1] * w[1];
    }
    void stiff_jacobian(double t, const std::vector<double>& w,
                        std::vector<double>& out) const {
      const double k = odelia::util::to_passive(K);
      out.assign(4, 0.0);
      out[0] = -2.0 * k * w[0];
      out[2] = 2.0 * k * w[0];
      out[3] = -2.0 * k * w[1];
    }
    void stiff_inputs(std::vector<double>& u) const { u = uptake; }
    void stiff_alone(double t, const std::vector<double>& w,
                     const std::vector<double>& u, std::vector<double>& out) const {
      ++store_alone_calls;
      stiff_rates(t, w, out);
      for (size_t i = 0; i < 2; ++i) {
        out[i] -= u[i];
      }
    }

  private:
    // A parameter at the scalar U: itself, or its value where U is double.
    template <class U> U at(const T& v) const {
      if constexpr (std::is_same_v<U, T>) {
        return v;
      } else {
        return odelia::util::to_passive(v);
      }
    }
    void compute_rates() {
      std::vector<T> w{w1, w2}, f(2);
      stiff_rates(time, w, f);
      const T u1 = c * x * w1, u2 = c * x * w2;
      uptake = {odelia::util::to_passive(u1), odelia::util::to_passive(u2)};
      dx = a * x * (w1 + w2) - m * x;
      dw1 = f[0] - u1;
      dw2 = f[1] - u2;
    }

    T a, m, K, s0, c;
    T x = 0, w1 = 0, w2 = 0, dx = 0, dw1 = 0, dw2 = 0;
    std::vector<double> uptake{0.0, 0.0};
    double time = 0;
  };
'

compile_ark_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = paste0('
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <cmath>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <examples/lorenz_system.hpp>
    ', store_system, '

    using namespace odelia;
    using StoreD = Store<double>;

    static_assert(ode::HasStiffBlock<StoreD>);
    static_assert(!ode::HasStiffBlock<LorenzSystem<double> >);

    ode::Method method_of(const std::string& name) {
      return name == "ark" ? ode::Method::ark : ode::Method::rkck;
    }

    ode::OdeControl control_at(double tol) {
      ode::OdeControl ctl;
      ctl.set_tol_abs(tol);
      ctl.set_tol_rel(tol);
      return ctl;
    }

    StoreD store(const std::vector<double>& p) {
      return StoreD(p[0], p[1], p[2], p[3], p[4]);
    }

    std::vector<ode::instruction> schedule_of(const std::vector<double>& pars,
                                              const std::vector<double>& y0,
                                              double t_end, double tol) {
      ode::Solver<StoreD> solver(store(pars), control_at(tol), ode::Method::ark);
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      return solver.schedule();
    }

    // [[Rcpp::export]]
    Rcpp::List store_run(std::string method, double tol, double t_end,
                         std::vector<double> pars, std::vector<double> y0) {
      store_alone_calls = 0;
      ode::Solver<StoreD> solver(store(pars), control_at(tol), method_of(method));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      return Rcpp::List::create(Rcpp::_["y"] = solver.state(),
                                Rcpp::_["steps"] = (int) solver.times().size() - 1,
                                Rcpp::_["alone_calls"] = store_alone_calls);
    }

    // The schedule an ARK run at `schedule_pars` took, replayed at `pars` from
    // `y0`: the sweep holds the sizes, so a difference of the run must too.
    // [[Rcpp::export]]
    Rcpp::List store_replay(std::vector<double> schedule_pars,
                            std::vector<double> pars, std::vector<double> y0,
                            std::vector<double> y0_schedule, double t_end,
                            double tol) {
      const std::vector<ode::instruction> schedule =
        schedule_of(schedule_pars, y0_schedule, t_end, tol);
      store_alone_calls = 0;
      ode::Solver<StoreD> replay(store(pars), control_at(tol), ode::Method::ark);
      replay.set_state(y0, 0.0);
      replay.advance_recorded(schedule);
      return Rcpp::List::create(Rcpp::_["y"] = replay.state(),
                                Rcpp::_["alone_calls"] = store_alone_calls);
    }

    // [[Rcpp::export]]
    Rcpp::List store_adjoint(std::vector<double> pars, std::vector<double> y0,
                             double t_end, std::vector<double> lambda_end,
                             double tol) {
      ode::Solver<StoreD> replay(store(pars), control_at(tol), ode::Method::ark);
      replay.set_keep_states(true);
      replay.set_state(y0, 0.0);
      replay.advance_recorded(schedule_of(pars, y0, t_end, tol));
      ode::adjoint_rows lambda = ode::adjoint_rows::one_row(lambda_end);
      ode::adjoint_rows rows(1, pars.size());
      replay.clear_recorded_rates();
      replay.solve_adjoint(lambda, rows);
      return Rcpp::List::create(
        Rcpp::_["lambda"] = lambda.to_rows()[0],
        Rcpp::_["parameter_adjoint"] = rows.to_rows()[0],
        Rcpp::_["n_steps"] = (int) replay.recording().size() - 1,
        Rcpp::_["recorded_rates"] = (int) replay.recorded_rates());
    }

    // One step from (t, y) of size h: the end, the pair embedded estimate, the
    // estimate block_error puts in its place, and the block inputs at both ends.
    // [[Rcpp::export]]
    Rcpp::List store_block_error(std::vector<double> pars, std::vector<double> y,
                                 double t, double h) {
      StoreD system = store(pars);
      ode::ArkStep<StoreD> stepper;
      stepper.resize(y.size());
      std::vector<double> dydt_in(y.size()), dydt_out(y.size()), yerr(y.size()),
        u0, u1;
      ode::derivs(system, y, dydt_in, t);
      system.stiff_inputs(u0);
      stepper.start_inputs(system);
      std::vector<double> y_end(y);
      ode::ArkStep<StoreD>::solved_row solved{};
      stepper.step(system, solved, t, h, y_end, yerr, dydt_in, dydt_out);
      const std::vector<double> embedded = yerr;
      system.stiff_inputs(u1);
      stepper.block_error(system, t, h, y, y_end, yerr);
      return Rcpp::List::create(Rcpp::_["y_end"] = y_end,
                                Rcpp::_["embedded"] = embedded,
                                Rcpp::_["yerr"] = yerr, Rcpp::_["u0"] = u0,
                                Rcpp::_["u1"] = u1);
    }

    // [[Rcpp::export]]
    void lorenz_ark() {
      LorenzSystem<double> lorenz(10.0, 28.0, 8.0 / 3.0);
      ode::Solver<LorenzSystem<double> > solver(lorenz, ode::OdeControl(),
                                                ode::Method::ark);
    }
  '), verbose = FALSE)
}

store_pars <- c(1.0, 0.8, 1e4, 200, 0.3)
store_y <- c(1.0, 0.5, 0.5)

testthat::test_that("ARK steps the stiff store in a fraction of Cash-Karp's steps, as accurately", {
  compile_ark_interface()

  reference <- store_run("rkck", 1e-12, 2.0, store_pars, store_y)$y
  ck <- store_run("rkck", 1e-6, 2.0, store_pars, store_y)
  ark <- store_run("ark", 1e-6, 2.0, store_pars, store_y)
  # Cash-Karp is held to its stability boundary by the store's drainage, so it
  # takes over a thousand steps; ARK steps the store implicitly.
  expect_gt(ck$steps, 1000L)
  expect_lt(ark$steps, ck$steps / 10)
  expect_lt(max(abs(ark$y - reference)), 1e-5)
  expect_lt(max(abs(ck$y - reference)), 1e-5)
  # It converges on the same answer as its tolerance tightens.
  tight <- store_run("ark", 1e-9, 2.0, store_pars, store_y)
  expect_lt(max(abs(tight$y - reference)), 1e-8)
})

testthat::test_that("a System that names no stiff block is refused at construction", {
  compile_ark_interface()
  expect_error(lorenz_ark(), "needs a System that names a stiff block")
})

testthat::test_that("the block's error estimate is the block alone's, and only an adaptive step forms it", {
  compile_ark_interface()

  t0 <- 0.3
  h <- 0.05
  # A state on the store quasi-equilibrium, as an adaptive run would hold.
  y <- c(0.8, 0.14, 0.14)
  r <- store_block_error(store_pars, y, t0, h)
  # The rest of the state keeps the pair's embedded estimate.
  expect_identical(r$yerr[1], r$embedded[1])

  # The block alone over the step, its uptake linear in time between the two
  # ends, by classical Runge-Kutta far finer than the estimate tolerance.
  f <- function(t, w) {
    inflow <- store_pars[4] * (1 + 0.5 * sin(2 * pi * t))
    k <- store_pars[3]
    c(inflow - k * w[1]^2, k * w[1]^2 - k * w[2]^2) -
      (r$u0 + (r$u1 - r$u0) * (t - t0) / h)
  }
  w <- y[2:3]
  n <- 4000
  dt <- h / n
  for (i in seq_len(n) - 1) {
    t <- t0 + i * dt
    k1 <- f(t, w)
    k2 <- f(t + dt / 2, w + dt / 2 * k1)
    k3 <- f(t + dt / 2, w + dt / 2 * k2)
    k4 <- f(t + dt, w + dt * k3)
    w <- w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  }
  expected <- r$y_end[2:3] - w
  expect_equal(r$yerr[2:3], expected, tolerance = 1e-6)
  # Which is not what the pair's embedded formula says.
  expect_gt(max(abs(r$embedded[2:3] - expected)),
            10 * max(abs(r$yerr[2:3] - expected)))

  # An adaptive run forms it at every attempt; a replay of that run never does.
  run <- store_run("ark", 1e-6, 2.0, store_pars, store_y)
  expect_gt(run$alone_calls, 0L)
  replay <- store_replay(store_pars, store_pars, store_y, store_y, 2.0, 1e-6)
  expect_identical(replay$alone_calls, 0L)
  # And the replay lands where the run did.
  expect_identical(replay$y, run$y)
})

testthat::test_that("the ARK sweep matches central differences of the replayed run", {
  compile_ark_interface()

  t_end <- 2.0
  tol <- 1e-6
  lambda_end <- c(0.4, -1.3, 0.7)
  r <- store_adjoint(store_pars, store_y, t_end, lambda_end, tol)
  # Six rate evaluations recorded per step, as Cash-Karp.
  expect_identical(r$recorded_rates, 6L * r$n_steps)

  # Relative perturbations, large enough that the run's rounding does not
  # swamp the smallest sensitivities.
  eps <- 1e-4
  replay <- function(pars, y0) {
    store_replay(store_pars, pars, y0, store_y, t_end, tol)$y
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
  # Each row against its own difference: the store's rows are thousands of times
  # smaller than the population row.
  expect_equal(r$lambda / expected_state, rep(1, 3), tolerance = 1e-5)
  # K and s0 are the stiff rates' own, so their rows pass through the implicit
  # function theorem at every implicit stage.
  expect_equal(r$parameter_adjoint / expected_pars, rep(1, 5), tolerance = 1e-5)
})
