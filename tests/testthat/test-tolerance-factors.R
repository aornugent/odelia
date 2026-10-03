# Tests that a System's tolerance factors decide the adaptive steps:
# factors of one reproduce the unscaled run bit for bit, a factor on one
# component changes the steps, and the factors are read at each step's start.

compile_state_tolerance_factors_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <examples/lorenz_system.hpp>

    using namespace odelia;
    using LorenzD = LorenzSystem<double>;

    // Lorenz with a factor on each component, multiplied by `late` on a step
    // starting at or after `from`.
    struct ScaledLorenz : LorenzD {
      using LorenzD::LorenzD;
      std::vector<double> factors;
      double from = 0.0, late = 1.0;
      void state_tolerance_factors(double time, std::vector<double>& f) const {
        f = factors;
        if (time >= from) {
          for (double& x : f) x *= late;
        }
      }
    };

    template <class System>
    Rcpp::List run(System sys, double t_end) {
      const std::vector<double> y0{1.0, 1.0, 1.0};
      sys.set_initial_state(y0.begin(), 0.0);
      sys.reset();
      ode::Solver<System> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      solver.advance_adaptive({0.0, t_end});
      return Rcpp::List::create(Rcpp::Named("times") = solver.times(),
                                Rcpp::Named("step_sizes") = solver.step_sizes(),
                                Rcpp::Named("state") = solver.state());
    }

    // [[Rcpp::export]]
    Rcpp::List lorenz_unscaled(double t_end) {
      return run(LorenzD(10.0, 28.0, 8.0 / 3.0), t_end);
    }

    // [[Rcpp::export]]
    Rcpp::List lorenz_scaled(std::vector<double> factors, double from,
                             double late, double t_end) {
      ScaledLorenz sys(10.0, 28.0, 8.0 / 3.0);
      sys.factors = factors;
      sys.from = from;
      sys.late = late;
      return run(sys, t_end);
    }
  ', verbose = FALSE)
}

test_that("factors of one reproduce the unscaled steps bit for bit", {
  compile_state_tolerance_factors_interface()
  expect_identical(lorenz_scaled(c(1, 1, 1), 0, 1, 2), lorenz_unscaled(2))
})

test_that("a factor on one component changes the steps", {
  compile_state_tolerance_factors_interface()
  plain <- lorenz_unscaled(2)
  tight <- lorenz_scaled(c(1, 1, 0.01), 0, 1, 2)
  loose <- lorenz_scaled(c(1, 1, 100), 0, 1, 2)
  expect_gt(length(tight$times), length(plain$times))
  expect_false(identical(loose$times, plain$times))
})

test_that("the factors are read at each step's start", {
  compile_state_tolerance_factors_interface()
  plain <- lorenz_unscaled(2)
  late <- lorenz_scaled(c(1, 1, 1), 1, 100, 2)
  # Every step starting before t = 1 is scaled by one, so the run matches the
  # unscaled one through the first step to end past 1, and leaves it after.
  n <- sum(plain$times < 1) + 1
  expect_identical(head(late$times, n), head(plain$times, n))
  expect_identical(head(late$step_sizes, n), head(plain$step_sizes, n))
  expect_false(identical(late$times, plain$times))
})
