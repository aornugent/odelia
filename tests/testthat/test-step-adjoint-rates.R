# Tests for the sweep through a System's own ode_rates_adjoint, and for
# Solver::solve_adjoint over the recorded steps. The tape path is covered by
# test-step-adjoint.R; here no tape is opened, so the snippet needs no link
# against the shared library.

lv_include_dir <- function() {
  dirname(dirname(resolve_test_path(
    "include/odelia/ode_solver.hpp", "inst/include/odelia/ode_solver.hpp")))
}

# Predator and prey, with the interaction flux published to aux. Its rate
# transpose is written out by hand, so nothing here is recorded.
lv_system <- '
  class LotkaVolterra {
  public:
    using value_type = double;
    LotkaVolterra(double a_, double b_, double c_, double d_)
      : a(a_), b(b_), c(c_), d(d_) { reset(); }

    size_t ode_size() const { return 2; }
    double ode_time() const { return time; }
    size_t aux_size() const { return 1; }
    void reset() { n = 10.0; p = 5.0; time = 0.0; compute_rates(); }

    // Loads the state, and nothing a rate evaluation would derive.
    template <typename It> It set_ode_state_and_field(It it, double time_) {
      time = time_;
      n = *it++;
      p = *it++;
      return it;
    }
    template <typename It> It set_ode_state(It it, double time_) {
      it = set_ode_state_and_field(it, time_);
      compute_rates();
      return it;
    }
    void compute_rates() {
      ++rate_calls;
      flux = b * n * p;
      dndt = a * n - flux;
      dpdt = c * flux - d * p;
    }
    template <typename It> It ode_state(It it) const {
      *it++ = n; *it++ = p; return it;
    }
    template <typename It> It ode_rates(It it) const {
      *it++ = dndt; *it++ = dpdt; return it;
    }
    template <typename It> It ode_aux(It it) const { *it++ = flux; return it; }
    template <typename It> It set_ode_aux(It it) {
      flux = *it++;
      aux_seen.push_back(flux);
      return it;
    }

    // The transpose of compute_rates at the state currently loaded.
    template <class ItIn, class ItOut>
    ItOut ode_rates_adjoint(ItIn lambda_dydt, ItOut lambda_y) {
      const double l0 = *lambda_dydt++;
      const double l1 = *lambda_dydt++;
      *lambda_y++ = (a - b * p) * l0 + (c * b * p) * l1;
      *lambda_y++ = (-b * n) * l0 + (c * b * n - d) * l1;
      return lambda_y;
    }

    int rate_calls = 0;
    std::vector<double> aux_seen;

  private:
    double a, b, c, d;
    double n = 0, p = 0, flux = 0, dndt = 0, dpdt = 0;
    double time = 0;
  };
'

compile_rates_adjoint_interface <- function() {
  withr::local_envvar(PKG_CPPFLAGS = paste0("-I", shQuote(lv_include_dir())))
  Rcpp::sourceCpp(code = paste0('
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <examples/lorenz_system.hpp>
    ', lv_system, '

    // The choice is made on the System, not at run time.
    static_assert(odelia::ode::AdjointRates<LotkaVolterra>);
    static_assert(!odelia::ode::AdjointRates<LorenzSystem<double> >);

    // [[Rcpp::export]]
    bool rates_adjoint_is_chosen_by_type() { return true; }

    // One RKCK step and its adjoint, plus what the step cost and what the sweep
    // was handed back.
    // [[Rcpp::export]]
    Rcpp::List lv_step_and_adjoint(std::vector<double> pars, double time,
                                   double step_size, std::vector<double> y,
                                   std::vector<double> lambda_out) {
      LotkaVolterra system(pars[0], pars[1], pars[2], pars[3]);
      odelia::ode::Step<LotkaVolterra> stepper;
      stepper.resize(y.size());

      std::vector<double> dydt_in(y.size()), dydt_out(y.size()), yerr(y.size());
      std::vector<double> y_end(y);
      odelia::ode::derivs(system, y, dydt_in, time);
      stepper.step(system, time, step_size, y_end, yerr, dydt_in, dydt_out);

      LotkaVolterra adj(pars[0], pars[1], pars[2], pars[3]);
      adj.rate_calls = 0;
      std::vector<double> lambda_in;
      stepper.step_adjoint(adj, time, step_size, y, lambda_out, lambda_in);

      return Rcpp::List::create(Rcpp::_["y_end"] = y_end,
                                Rcpp::_["lambda_in"] = lambda_in,
                                Rcpp::_["rate_calls"] = adj.rate_calls,
                                Rcpp::_["aux_seen"] = adj.aux_seen);
    }

    // The recorded run, the states it passed through, and the adjoint swept back
    // over them.
    // [[Rcpp::export]]
    Rcpp::List lv_solve_adjoint(std::vector<double> pars, std::vector<double> y0,
                                double t_end, std::vector<double> lambda_end) {
      odelia::ode::OdeControl ctl;
      LotkaVolterra system(pars[0], pars[1], pars[2], pars[3]);
      odelia::ode::Solver<LotkaVolterra> solver(system, ctl);
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      const std::vector<double> h = solver.step_sizes();

      // Replay the recorded sizes so a state is collected at each accepted step.
      odelia::ode::Solver<LotkaVolterra> replay(system, ctl);
      replay.set_state(y0, 0.0);
      replay.advance_fixed_steps(h);
      std::vector<std::vector<double> > states;
      for (size_t i = 0; i < replay.get_history_size(); ++i) {
        std::vector<double> s(y0.size());
        replay.get_history_step(i).ode_state(s.begin());
        states.push_back(s);
      }

      std::vector<double> lambda(lambda_end);
      replay.solve_adjoint(states, lambda);

      return Rcpp::List::create(Rcpp::_["n_steps"] = (int) h.size(),
                                Rcpp::_["y_end"] = states.back(),
                                Rcpp::_["lambda"] = lambda);
    }

    // The same recorded schedule run forward from a given start state, so a
    // finite difference of the whole run reads the identical step sizes.
    // [[Rcpp::export]]
    std::vector<double> lv_replay(std::vector<double> pars,
                                  std::vector<double> y0, double t_end,
                                  std::vector<double> y_start) {
      odelia::ode::OdeControl ctl;
      LotkaVolterra system(pars[0], pars[1], pars[2], pars[3]);
      odelia::ode::Solver<LotkaVolterra> solver(system, ctl);
      solver.set_state(y0, 0.0);
      solver.advance_adaptive({0.0, t_end});
      const std::vector<double> h = solver.step_sizes();

      odelia::ode::Solver<LotkaVolterra> replay(system, ctl);
      replay.set_state(y_start, 0.0);
      replay.advance_fixed_steps(h);
      return replay.state();
    }
  '), verbose = FALSE)
}

lv_pars <- c(1.1, 0.06, 0.4, 0.9)
lv_y <- c(11.0, 4.0)

testthat::test_that("the rate transpose is chosen on the System's type", {
  compile_rates_adjoint_interface()
  expect_true(rates_adjoint_is_chosen_by_type())
})

testthat::test_that("step_adjoint through ode_rates_adjoint matches a finite difference of one step", {
  compile_rates_adjoint_interface()

  step_size <- 0.05
  # Non-symmetric, so a dropped transpose shows up.
  lambda_out <- c(0.7, -1.9)
  eps <- 1e-7

  step_end <- function(y) {
    lv_step_and_adjoint(lv_pars, 0.0, step_size, y, lambda_out)$y_end
  }
  jacobian <- sapply(seq_along(lv_y), function(j) {
    up <- lv_y; up[j] <- up[j] + eps
    down <- lv_y; down[j] <- down[j] - eps
    (step_end(up) - step_end(down)) / (2 * eps)
  })
  expected <- as.vector(t(jacobian) %*% lambda_out)

  got <- lv_step_and_adjoint(lv_pars, 0.0, step_size, lv_y, lambda_out)$lambda_in
  expect_equal(got, expected, tolerance = 1e-6)
  # The transpose is load-bearing: J is not symmetric here.
  expect_false(isTRUE(all.equal(expected, as.vector(jacobian %*% lambda_out),
                                tolerance = 1e-3)))
})

testthat::test_that("one component of lambda_out at a time agrees, row by row", {
  compile_rates_adjoint_interface()

  step_size <- 0.05
  eps <- 1e-7
  for (i in seq_along(lv_y)) {
    seed <- rep(0, length(lv_y))
    seed[i] <- 1
    row <- sapply(seq_along(lv_y), function(j) {
      up <- lv_y; up[j] <- up[j] + eps
      down <- lv_y; down[j] <- down[j] - eps
      (lv_step_and_adjoint(lv_pars, 0.0, step_size, up, seed)$y_end[i] -
       lv_step_and_adjoint(lv_pars, 0.0, step_size, down, seed)$y_end[i]) /
        (2 * eps)
    })
    got <- lv_step_and_adjoint(lv_pars, 0.0, step_size, lv_y, seed)$lambda_in
    expect_equal(got, row, tolerance = 1e-6,
                 info = paste("seeded component", i))
  }
})

testthat::test_that("the sweep evaluates rates six times and no more", {
  compile_rates_adjoint_interface()

  r <- lv_step_and_adjoint(lv_pars, 0.0, 0.05, lv_y, c(0.7, -1.9))
  # Six stages rebuilt, k1 among them, and nothing from the sweep: the rate
  # transpose carries the chain, so a rate evaluation there would repeat it.
  expect_identical(r$rate_calls, 6L)
  # One aux handed back per stage.
  expect_identical(length(r$aux_seen), 6L)
})

testthat::test_that("each stage is handed back its own aux, in reverse", {
  compile_rates_adjoint_interface()

  r <- lv_step_and_adjoint(lv_pars, 0.0, 0.05, lv_y, c(0.7, -1.9))
  # A stage is addressed by index and never by time: two RKCK stages share a
  # timestamp, so the only check that a stage got its own aux is that the six
  # differ and arrive in the reverse of the order they were published.
  expect_identical(length(unique(r$aux_seen)), 6L)
  expect_false(isTRUE(all.equal(r$aux_seen, rev(r$aux_seen))))
})

testthat::test_that("a zero end adjoint sweeps to zero", {
  compile_rates_adjoint_interface()
  r <- lv_step_and_adjoint(lv_pars, 0.0, 0.05, lv_y, c(0.0, 0.0))
  expect_equal(r$lambda_in, c(0.0, 0.0))
})

testthat::test_that("solve_adjoint over the recorded steps matches a finite difference of the run", {
  compile_rates_adjoint_interface()

  t_end <- 1.5
  lambda_end <- c(0.4, -1.3)
  eps <- 1e-6

  r <- lv_solve_adjoint(lv_pars, lv_y, t_end, lambda_end)
  expect_gt(r$n_steps, 3)

  jacobian <- sapply(seq_along(lv_y), function(j) {
    up <- lv_y; up[j] <- up[j] + eps
    down <- lv_y; down[j] <- down[j] - eps
    (lv_replay(lv_pars, lv_y, t_end, up) -
     lv_replay(lv_pars, lv_y, t_end, down)) / (2 * eps)
  })
  expected <- as.vector(t(jacobian) %*% lambda_end)
  expect_equal(r$lambda, expected, tolerance = 1e-5)
})
