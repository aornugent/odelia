# Tests that a System's tolerance scales decide the adaptive steps: scales of
# one reproduce the unscaled run bit for bit, a scale multiplies its
# component's error level under both step-size rules, and the scales are read
# at each step's start.

compile_tolerance_scales_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <string>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <examples/lorenz_system.hpp>

    using namespace odelia;
    using LorenzD = LorenzSystem<double>;

    // Lorenz with a scale on each component, multiplied by `late` on a step
    // starting at or after `from`.
    struct ScaledLorenz : LorenzD {
      using LorenzD::LorenzD;
      std::vector<double> scales;
      double from = 0.0, late = 1.0;
      void tolerance_scales(double time, std::vector<double>& f) const {
        f = scales;
        if (time >= from) {
          for (double& x : f) x *= late;
        }
      }
    };

    static_assert(ode::ScalesTolerances<ScaledLorenz>);
    static_assert(!ode::ScalesTolerances<LorenzD>);

    template <class System>
    Rcpp::List run(System sys, double t_end, double tol,
                   const std::string& controller) {
      const std::vector<double> y0{1.0, 1.0, 1.0};
      sys.set_initial_state(y0.begin(), 0.0);
      sys.reset();
      ode::OdeControl control;
      control.set_tol_abs(tol);
      control.set_tol_rel(tol);
      control.set_controller(controller == "gsl" ? ode::Controller::gsl
                                                 : ode::Controller::hairer);
      ode::Solver<System> solver(sys, control);
      solver.set_collect(false);
      solver.advance_adaptive({0.0, t_end});
      return Rcpp::List::create(Rcpp::Named("times") = solver.times(),
                                Rcpp::Named("state") = solver.state());
    }

    // [[Rcpp::export]]
    Rcpp::List lorenz_unscaled(double t_end, double tol = 1e-8,
                               std::string controller = "hairer") {
      return run(LorenzD(10.0, 28.0, 8.0 / 3.0), t_end, tol, controller);
    }

    // [[Rcpp::export]]
    Rcpp::List lorenz_scaled(std::vector<double> scales, double from,
                             double late, double t_end, double tol = 1e-8,
                             std::string controller = "hairer") {
      ScaledLorenz sys(10.0, 28.0, 8.0 / 3.0);
      sys.scales = scales;
      sys.from = from;
      sys.late = late;
      return run(sys, t_end, tol, controller);
    }
  ', verbose = FALSE)
}

test_that("scales of one reproduce the unscaled steps bit for bit", {
  compile_tolerance_scales_interface()
  for (controller in c("hairer", "gsl")) {
    expect_identical(lorenz_scaled(c(1, 1, 1), 0, 1, 2, controller = controller),
                     lorenz_unscaled(2, controller = controller))
  }
})

test_that("a scale of 4 on every component is the tolerance times 4, under both rules", {
  # A power of two scales each error level exactly, so the two runs agree to the
  # bit only if the scale reaches every level each rule reads.
  compile_tolerance_scales_interface()
  for (controller in c("hairer", "gsl")) {
    expect_identical(
      lorenz_scaled(c(4, 4, 4), 0, 1, 2, tol = 1e-8, controller = controller),
      lorenz_unscaled(2, tol = 4e-8, controller = controller))
  }
})

test_that("a scale on one component changes the steps", {
  compile_tolerance_scales_interface()
  plain <- lorenz_unscaled(2)
  tight <- lorenz_scaled(c(1, 1, 0.01), 0, 1, 2)
  loose <- lorenz_scaled(c(1, 1, 100), 0, 1, 2)
  expect_gt(length(tight$times), length(plain$times))
  expect_false(identical(loose$times, plain$times))
})

test_that("the scales are read at each step's start", {
  compile_tolerance_scales_interface()
  plain <- lorenz_unscaled(2)
  later <- lorenz_scaled(c(1, 1, 1), 1, 100, 2)
  # Every step that starts before t = 1 is the unscaled run's.
  before <- plain$times[plain$times <= 1]
  n <- length(before) + 1
  expect_identical(later$times[seq_len(n)], plain$times[seq_len(n)])
  expect_lt(length(later$times), length(plain$times))
})

test_that("a scale that is not positive and finite, or one too few, is refused", {
  compile_tolerance_scales_interface()
  expect_error(lorenz_scaled(c(1, 0, 1), 0, 1, 2), "positive and finite")
  expect_error(lorenz_scaled(c(1, Inf, 1), 0, 1, 2), "positive and finite")
  expect_error(lorenz_scaled(c(1, NaN, 1), 0, 1, 2), "positive and finite")
  expect_error(lorenz_scaled(c(1, 1), 0, 1, 2), "[Ii]ncorrect length")
})
