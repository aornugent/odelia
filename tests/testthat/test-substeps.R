# Tests for a step that substeps a System's subsystem: where a stiff subsystem
# reads the rest only through a few inputs, the rest takes long steps while the
# subsystem takes substeps of its own, which replays and the sweep take as
# recorded.

# A population x drawing on a two-layer store w, which a seasonal source fills
# and drainage K w^2 carries layer to layer: stiff in w where K is large. The
# uptake c x w is what the rest of the state hands the store. With
# `WithSubsystem` false it is the same System with no subsystem.
store_system <- '
  template <typename T, bool WithSubsystem = true>
  class Store {
  public:
    using value_type = T;
    Store(T a_ = T(0), T m_ = T(0), T K_ = T(0), T s0_ = T(0), T c_ = T(0))
      : a(a_), m(m_), K(K_), s0(s0_), c(c_) { reset(); }

    template <typename, bool> friend class Store;

    template <class S2>
    void assign_from(const Store<S2, WithSubsystem>& src) {
      a = T(odelia::util::to_passive(src.a));
      m = T(odelia::util::to_passive(src.m));
      K = T(odelia::util::to_passive(src.K));
      s0 = T(odelia::util::to_passive(src.s0));
      c = T(odelia::util::to_passive(src.c));
      x = T(odelia::util::to_passive(src.x));
      w1 = T(odelia::util::to_passive(src.w1));
      w2 = T(odelia::util::to_passive(src.w2));
      time = src.time;
      fail_after = src.fail_after;
      pulses = src.pulses;
      compute_rates();
    }

    template <class S2>
    Store<S2, WithSubsystem> rebind() const {
      Store<S2, WithSubsystem> out;
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
    template <class V> void set_recorded_state(const V& y, double time_) {
      set_ode_state(y.begin(), time_);
    }
    template <typename It> It ode_state(It it) const {
      *it++ = x; *it++ = w1; *it++ = w2;
      return it;
    }
    template <typename It> It ode_rates(It it) const {
      *it++ = dx; *it++ = dw1; *it++ = dw2;
      return it;
    }

    // The store, from index 1.
    odelia::ode::state_range subsystem() const requires WithSubsystem {
      return {1, 2};
    }
    std::vector<double> subsystem_knots(double from, double to) const
      requires WithSubsystem {
      return pulses != nullptr ? pulses->knots(from, to) : std::vector<double>{};
    }
    void subsystem_inputs(std::vector<T>& u) const requires WithSubsystem {
      u = {u1, u2};
    }
    template <class U>
    void subsystem_rates(double t, const std::vector<U>& w,
                         const std::vector<U>& u,
                         std::vector<U>& out) const requires WithSubsystem {
      store_rates(t, w, out);
      out[0] -= u[0];
      out[1] -= u[1];
      if (t > fail_after) {
        out[0] = U(std::nan(""));
      }
    }

    // Past this time the store has NaN rates.
    double fail_after = 1e300;
    // Pulses of inflow on top of the seasonal one.
    const odelia::drivers::Function* pulses = nullptr;

  private:
    template <class U> U at(const T& v) const {
      if constexpr (std::is_same_v<U, T>) {
        return v;
      } else {
        return odelia::util::to_passive(v);
      }
    }
    double pulse_at(double t) const {
      return pulses != nullptr ? std::max(0.0, pulses->evaluate(t)) : 0.0;
    }
    T inflow_at(double t) const {
      return s0 * (1.0 + 0.5 * std::sin(2.0 * M_PI * t)) + pulse_at(t);
    }
    template <class U>
    void store_rates(double t, const std::vector<U>& w, std::vector<U>& out) const {
      const U k = at<U>(K);
      const U inflow = at<U>(s0) * (1.0 + 0.5 * std::sin(2.0 * M_PI * t)) + pulse_at(t);
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

compile_substep_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  # Without contraction, because the tests below hold a replay at another
  # scalar to the run bit for bit (see compile_step_adjoint_interface).
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_CXXFLAGS = "-ffp-contract=off",
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = paste0('
    // [[Rcpp::plugins(cpp20)]]
    // Built with -ffp-contract=off (see compile_substep_interface).
    #include <Rcpp.h>
    #include <algorithm>
    #include <cmath>
    #include <limits>
    #include <vector>
    #include <odelia/drivers.hpp>
    #include <odelia/ode_solver.hpp>
    #include <odelia/tangent.hpp>
    ', store_system, '

    using namespace odelia;
    using StoreA = Store<double, true>;
    using StoreP = Store<double, false>;

    static_assert(ode::HasSubsystem<StoreA>);
    static_assert(!ode::HasSubsystem<StoreP>);

    // The store with a tolerance scale on its two layers.
    struct ScaledStore : StoreA {
      using StoreA::StoreA;
      double store_scale = 1.0;
      void tolerance_scales(double, std::vector<double>& scales) const {
        scales = {1.0, store_scale, store_scale};
      }
    };

    // The GSL rule, which plant runs under: the Hairer rule grows a step across
    // more of the store pulses than the population stages resolve.
    ode::OdeControl control_at(double tol, double substep_tol = 1e-9) {
      ode::OdeControl ctl;
      ctl.set_tol_abs(tol);
      ctl.set_tol_rel(tol);
      ctl.set_controller(ode::Controller::gsl);
      ctl.substep_tol = substep_tol;
      return ctl;
    }

    StoreA store(const std::vector<double>& p) {
      StoreA s(p[0], p[1], p[2], p[3], p[4]);
      return s;
    }

    std::vector<ode::schedule_row> schedule_of(const std::vector<double>& pars,
                                             const std::vector<double>& y0,
                                             double t_end, double tol) {
      ode::Solver<StoreA> solver(store(pars), control_at(tol));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      return solver.steps_taken();
    }

    // [[Rcpp::export]]
    Rcpp::List store_run(double tol, double t_end,
                         std::vector<double> pars, std::vector<double> y0,
                         double fail_after = 1e300, double substep_tol = 1e-9) {
      const std::vector<double> fractions{1.0 / 5.0, 0.3, 3.0 / 5.0, 7.0 / 8.0,
                                          1.0};
      StoreA s = store(pars);
      s.fail_after = fail_after;
      ode::Solver<StoreA> solver(s, control_at(tol, substep_tol));
      solver.set_keep_states(true);
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      const std::vector<ode::schedule_row> schedule = solver.steps_taken();
      int substepped = 0, substeps = 0;
      bool fractions_landed = true, ascending = true, sized = true;
      for (const ode::schedule_row& row : schedule) {
        if (!row.subsystem) {
          continue;
        }
        const ode::subsystem_substeps& taken = *row.subsystem;
        ++substepped;
        substeps += taken.ends.size();
        sized = sized && taken.input_slope.size() == 2;
        ascending = ascending && std::is_sorted(taken.ends.begin(),
                                                taken.ends.end());
        for (double fraction : fractions) {
          fractions_landed = fractions_landed &&
            std::find(taken.ends.begin(), taken.ends.end(), fraction) !=
              taken.ends.end();
        }
      }
      // A replay of the recording, its rows loaded, repeats the run.
      ode::Solver<StoreA> replay(store(pars), control_at(tol));
      replay.set_state(y0, 0.0);
      replay.replay(solver.recording());
      return Rcpp::List::create(
        Rcpp::_["y"] = solver.state(),
        Rcpp::_["steps"] = (int) schedule.size() - 1,
        Rcpp::_["substepped"] = substepped, Rcpp::_["substeps"] = substeps,
        Rcpp::_["fractions_landed"] = fractions_landed,
        Rcpp::_["ascending"] = ascending,
        Rcpp::_["sized"] = sized,
        Rcpp::_["recording_replayed"] = replay.state());
    }

    // A run under pulses of inflow through (x, y), stopping at `stops`: where it
    // lands, the times its steps end, whether each step substepped the store, and
    // the times its substeps end.
    // [[Rcpp::export]]
    Rcpp::List store_pulsed_run(double tol, std::vector<double> stops,
                                std::vector<double> pars, std::vector<double> y0,
                                std::vector<double> x, std::vector<double> y) {
      const drivers::Function pulses(x, y, drivers::Slopes::monotone);
      StoreA s = store(pars);
      s.pulses = &pulses;
      ode::Solver<StoreA> solver(s, control_at(tol));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive(stops);
      const std::vector<ode::schedule_row> schedule = solver.steps_taken();
      std::vector<double> times, substep_ends;
      std::vector<bool> substepped;
      for (size_t k = 0; k < schedule.size(); ++k) {
        times.push_back(schedule[k].time);
        if (k > 0) {
          substepped.push_back(schedule[k].subsystem.has_value());
        }
        if (k > 0 && schedule[k].subsystem) {
          for (double end : schedule[k].subsystem->ends) {
            substep_ends.push_back(schedule[k - 1].time + end * schedule[k].step_size);
          }
        }
      }
      return Rcpp::List::create(Rcpp::_["y"] = solver.state(),
                                Rcpp::_["steps"] = (int) schedule.size() - 1,
                                Rcpp::_["times"] = times,
                                Rcpp::_["substepped"] = substepped,
                                Rcpp::_["substep_ends"] = substep_ends);
    }

    // The schedule of a pulsed run, replayed with its substeps dropped, which a step
    // spanning a knot refuses.
    // [[Rcpp::export]]
    void store_pulsed_replay_without_substeps(double tol,
                                              std::vector<double> pars,
                                              std::vector<double> y0,
                                              std::vector<double> x,
                                              std::vector<double> y) {
      const drivers::Function pulses(x, y, drivers::Slopes::monotone);
      StoreA s = store(pars);
      s.pulses = &pulses;
      ode::Solver<StoreA> run(s, control_at(tol));
      run.set_state(y0, 0.0);
      run.advance_adaptive({0.0, 1.0});
      std::vector<ode::schedule_row> schedule = run.steps_taken();
      for (ode::schedule_row& row : schedule) {
        row.subsystem.reset();
      }
      ode::Solver<StoreA> replay(s, control_at(tol));
      replay.set_state(y0, 0.0);
      replay.advance_schedule(schedule);
    }

    // A run with the tolerance on the store scaled by `store_scale`: where it
    // lands, its steps and substepped steps, and where the unscaled store lands
    // replaying them.
    // [[Rcpp::export]]
    Rcpp::List store_scaled_run(double tol, double t_end,
                                std::vector<double> pars,
                                std::vector<double> y0, double store_scale) {
      ScaledStore s(pars[0], pars[1], pars[2], pars[3], pars[4]);
      s.store_scale = store_scale;
      ode::Solver<ScaledStore> solver(s, control_at(tol));
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      const std::vector<ode::schedule_row> schedule = solver.steps_taken();
      int substepped = 0;
      for (const ode::schedule_row& row : schedule) {
        substepped += row.subsystem.has_value();
      }
      ode::Solver<StoreA> replay(store(pars), control_at(tol));
      replay.set_state(y0, 0.0);
      replay.advance_schedule(schedule);
      return Rcpp::List::create(
        Rcpp::_["y"] = solver.state(),
        Rcpp::_["steps"] = (int) schedule.size() - 1,
        Rcpp::_["substepped"] = substepped,
        Rcpp::_["schedule_replayed"] = replay.state());
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

    // The schedule a run at `schedule_pars` took, replayed at `pars` from `y0`:
    // where it lands, and whether its own rows took the substeps the run took.
    // [[Rcpp::export]]
    Rcpp::List store_replay(std::vector<double> schedule_pars,
                            std::vector<double> pars, std::vector<double> y0,
                            std::vector<double> y0_schedule, double t_end,
                            double tol, std::string method) {
      const std::vector<ode::schedule_row> schedule =
        schedule_of(schedule_pars, y0_schedule, t_end, tol);
      ode::Solver<StoreA> replay(store(pars), control_at(tol),
                                 method == "rodas" ? ode::Method::rodas
                                                   : ode::Method::rkck);
      replay.set_state(y0, 0.0);
      replay.advance_schedule(schedule);
      const std::vector<ode::schedule_row> taken = replay.steps_taken();
      bool same_steps = taken.size() == schedule.size();
      for (size_t k = 0; same_steps && k < taken.size(); ++k) {
        same_steps = taken[k].subsystem == schedule[k].subsystem;
      }
      return Rcpp::List::create(Rcpp::_["y"] = replay.state(),
                                Rcpp::_["same_steps"] = same_steps);
    }

    // The schedule a run took, replayed at the tangent scalar along `direction`
    // in the parameters: where it lands, and its derivative there.
    // [[Rcpp::export]]
    Rcpp::List store_tangent_replay(std::vector<double> pars, std::vector<double> y0,
                                    double t_end, double tol,
                                    std::vector<double> direction) {
      const std::vector<ode::schedule_row> schedule =
        schedule_of(pars, y0, t_end, tol);
      using tangent = ode::tangent_scalar<>;
      auto active = store(pars).rebind<tangent>();
      std::vector<tangent*> p = active.ad_parameters();
      for (size_t j = 0; j < p.size(); ++j) {
        ode::set_direction(*p[j], direction[j]);
      }
      ode::Solver<Store<tangent, true>> replay(active, control_at(tol));
      replay.set_collect(false);
      replay.set_state(y0, 0.0);
      replay.advance_schedule(schedule);
      std::vector<double> y, dy;
      for (const tangent& v : replay.state()) {
        y.push_back(xad::value(v));
        dy.push_back(ode::derivative_along(v));
      }
      return Rcpp::List::create(Rcpp::_["y"] = y, Rcpp::_["dy"] = dy);
    }

    // The times a run took, replayed as a grid, which a step that would
    // substep the store refuses.
    // [[Rcpp::export]]
    Rcpp::List store_grid_replay(std::vector<double> pars, std::vector<double> y0,
                                 double t_end, double tol) {
      std::vector<ode::schedule_row> grid = schedule_of(pars, y0, t_end, tol);
      for (ode::schedule_row& row : grid) {
        row = {row.time, std::numeric_limits<double>::quiet_NaN()};
      }
      ode::Solver<StoreA> replay(store(pars), control_at(tol));
      replay.set_state(y0, 0.0);
      replay.advance_schedule(grid);
      return Rcpp::List::create(Rcpp::_["y"] = replay.state());
    }

    // [[Rcpp::export]]
    Rcpp::List store_adjoint(std::vector<double> pars, std::vector<double> y0,
                             double t_end, std::vector<double> lambda_end,
                             double tol) {
      ode::Solver<StoreA> run(store(pars), control_at(tol));
      run.set_keep_states(true);
      run.set_state(y0, 0.0);
      run.advance_adaptive({0.0, t_end});
      ode::adjoint_batch lambda = ode::adjoint_batch::one(lambda_end);
      ode::adjoint_batch rows(1, pars.size());
      run.clear_recorded_rates();
      run.sweep_adjoint(lambda, rows);
      return Rcpp::List::create(
        Rcpp::_["lambda"] = lambda.to_vectors()[0],
        Rcpp::_["parameter_adjoint"] = rows.to_vectors()[0],
        Rcpp::_["n_steps"] = (int) run.recording().size() - 1,
        Rcpp::_["recorded_rates"] = (int) run.recorded_rates());
    }
  '), verbose = FALSE)
}

store_pars <- c(1.0, 0.8, 1e4, 200, 0.3)
store_y <- c(1.0, 0.5, 0.5)
# Drainage K = 10: a store inside Cash-Karp's stability limit, whose coupling
# error exceeds Cash-Karp's own on it, so Cash-Karp's error never sets the step.
slow <- store_pars * c(1, 1, 1e-3, 1, 1)

testthat::test_that("where the store's Cash-Karp error never sets the step, every step is the plain System's", {
  compile_substep_interface()
  off <- store_run(1e-6, 0.5, slow, store_y)
  plain <- store_plain_run(1e-6, 0.5, slow, store_y)
  expect_identical(off$substepped, 0L)
  expect_identical(off$steps, plain$steps)
  expect_identical(off$y, plain$y)
})

testthat::test_that("substepped, the stiff store lets the population take long steps, as accurately", {
  compile_substep_interface()
  reference <- store_plain_run(1e-12, 2.0, store_pars, store_y)$y
  ck <- store_plain_run(1e-6, 2.0, store_pars, store_y)
  substepped <- store_run(1e-6, 2.0, store_pars, store_y)
  # Cash-Karp is held to its stability limit by the store's drainage; a step
  # past it takes substeps of the store instead.
  expect_gt(ck$steps, 1000L)
  expect_gt(substepped$substepped, substepped$steps / 2)
  expect_lt(substepped$steps, ck$steps / 10)
  expect_lt(max(abs(substepped$y - reference)), 1e-5)
  # It converges on the same answer as its tolerance tightens.
  tight <- store_run(1e-9, 2.0, store_pars, store_y)
  expect_lt(max(abs(tight$y - reference)), 1e-8)
})

testthat::test_that("a store the population takes nothing from keeps substepping", {
  compile_substep_interface()
  # With no uptake the predictor and the corrector end at the same state, which
  # measures no stiffness; the last measure stands.
  no_uptake <- store_pars * c(1, 1, 1, 1, 0)
  ck <- store_plain_run(1e-4, 2.0, no_uptake, store_y)
  substepped <- store_run(1e-4, 2.0, no_uptake, store_y)
  expect_lt(substepped$steps, ck$steps / 10)
})

testthat::test_that("each row records its input slope and substeps, landing on every stage fraction", {
  compile_substep_interface()
  r <- store_run(1e-6, 2.0, store_pars, store_y)
  expect_true(r$sized)
  expect_true(r$ascending)
  expect_true(r$fractions_landed)
  expect_gt(r$substeps, 5L * r$substepped)
  expect_identical(r$recording_replayed, r$y)
})

testthat::test_that("a store whose rates fail at every substep stops the run, saying why", {
  compile_substep_interface()
  expect_error(store_run(1e-6, 2.0, store_pars, store_y, fail_after = 1.0),
               "subsystem fails its error test at the smallest substep")
})

testthat::test_that("a replay takes the run's substeps, at the run's parameters to the bit", {
  compile_substep_interface()
  run <- store_run(1e-6, 2.0, store_pars, store_y)
  same <- store_replay(store_pars, store_pars, store_y, store_y, 2.0, 1e-6, "rkck")
  expect_identical(same$y, run$y)
  expect_true(same$same_steps)
  moved <- store_pars * c(1, 1, 1.01, 1, 1)
  away <- store_replay(store_pars, moved, store_y, store_y, 2.0, 1e-6, "rkck")
  expect_true(away$same_steps)
  expect_false(identical(away$y, run$y))
  expect_error(store_replay(store_pars, store_pars, store_y, store_y, 2.0, 1e-6,
                            "rodas"),
               "taken only by method='rkck'")
})

testthat::test_that("the substeps' tolerance is the control's", {
  compile_substep_interface()
  reference <- store_plain_run(1e-12, 2.0, store_pars, store_y)$y
  tight <- store_run(1e-6, 2.0, store_pars, store_y)
  loose <- store_run(1e-6, 2.0, store_pars, store_y, substep_tol = 1e-6)
  expect_identical(store_run(1e-6, 2.0, store_pars, store_y, substep_tol = 1e-9),
                   tight)
  expect_lt(loose$substeps, tight$substeps)
  expect_equal(loose$y, reference, tolerance = 1e-5)
})

testthat::test_that("a tolerance scale on the store decides its steps, and a replay repeats them", {
  compile_substep_interface()
  run <- store_run(1e-6, 2.0, store_pars, store_y)
  ones <- store_scaled_run(1e-6, 2.0, store_pars, store_y, 1)
  expect_identical(ones$y, run$y)
  expect_identical(ones$steps, run$steps)
  expect_identical(ones$substepped, run$substepped)
  loose <- store_scaled_run(1e-6, 2.0, store_pars, store_y, 100)
  expect_lt(loose$steps, run$steps)
  expect_identical(loose$schedule_replayed, loose$y)
})

testthat::test_that("a grid of a run's times is refused where a step would substep the store", {
  compile_substep_interface()
  expect_error(store_grid_replay(store_pars, store_y, 0.5, 1e-6),
               "step the run's schedule")
  # Where the store's Cash-Karp error never sets the step no step substeps, and
  # the grid is stepped.
  off <- store_run(1e-6, 0.5, slow, store_y)
  expect_equal(store_grid_replay(slow, store_y, 0.5, 1e-6)$y, off$y,
               tolerance = 1e-12)
})

testthat::test_that("a tangent replay takes the run's substeps, its value the run's to the bit", {
  compile_substep_interface()
  direction <- c(0, 0, 1, 0, 0)
  r <- store_tangent_replay(store_pars, store_y, 2.0, 1e-6, direction)
  run <- store_run(1e-6, 2.0, store_pars, store_y)
  expect_identical(r$y, run$y)
  e <- 1e-4 * store_pars[3]
  replay <- function(pars) {
    store_replay(store_pars, pars, store_y, store_y, 2.0, 1e-6, "rkck")$y
  }
  expected <- (replay(store_pars + e * direction) -
               replay(store_pars - e * direction)) / (2 * e)
  expect_equal(r$dy, expected, tolerance = 1e-5)
})

testthat::test_that("the sweep through substepped steps matches central differences of the replayed schedule", {
  compile_substep_interface()
  t_end <- 2.0
  tol <- 1e-6
  lambda_end <- c(0.4, -1.3, 0.7)
  r <- store_adjoint(store_pars, store_y, t_end, lambda_end, tol)
  expect_identical(r$recorded_rates, 6L * r$n_steps)

  eps <- 1e-4
  replay <- function(pars, y0) {
    store_replay(store_pars, pars, y0, store_y, t_end, tol, "rkck")$y
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

# Pulses of inflow two hundredths wide, every quarter of a unit, through a
# monotone cubic; its knots are those beside a non-zero value, where the cubic
# changes.
pulse_x <- seq(0, 2.5, by = 0.01)
pulse_y <- ifelse(seq_along(pulse_x) %% 25 == 5, 300, 0)
pulse_knots <- local({
  nz <- pulse_y != 0
  n <- length(nz)
  pulse_x[nz | c(FALSE, nz[-n]) | c(nz[-1], FALSE)]
})

testthat::test_that("a step spanning the inflow's knots substeps the store, landing a substep on each", {
  compile_substep_interface()
  t_end <- 1.0
  knots <- pulse_knots[pulse_knots < t_end]
  # The slow store holds each pulse as a rise in its level.
  reference <- store_pulsed_run(1e-11, c(0, knots, t_end), slow, store_y,
                                pulse_x, pulse_y)$y
  stopped <- store_pulsed_run(1e-6, c(0, knots, t_end), slow, store_y,
                              pulse_x, pulse_y)
  free <- store_pulsed_run(1e-6, c(0, t_end), slow, store_y,
                           pulse_x, pulse_y)
  # Stopped at every knot, no step spans one and none substeps the store.
  expect_false(any(stopped$substepped))
  t <- free$times
  spans <- vapply(seq_along(free$substepped), function(k)
    any(knots > t[k] & knots < t[k + 1]), TRUE)
  expect_true(any(spans))
  expect_true(all(free$substepped[spans]))
  landed <- vapply(knots, function(k) any(abs(c(t, free$substep_ends) - k) < 1e-12), TRUE)
  expect_true(all(landed))
  # Within the tolerance in the solver's norm, |error| / (1 + |y|).
  error <- function(y) max(abs(y - reference) / (1 + abs(reference)))
  expect_lt(error(stopped$y), 1e-6)
  expect_lt(error(free$y), 1e-6)
  # A replay that lost those substeps is refused at the first step across a knot.
  expect_error(store_pulsed_replay_without_substeps(1e-6, slow, store_y,
                                                    pulse_x, pulse_y),
               "carries no substeps")
})
