# Tests for odelia::implicit_value -- the value defined implicitly by a scalar
# equation, made differentiable through the implicit function theorem -- and for the
# adaptive controller's rejection of a non-finite step-size decision, which is what a
# node that goes non-finite reaches next.
#
# implicit_value records on a tape, and the XAD Tape<T,N> template methods are
# explicitly instantiated only in the odelia shared library, so these snippets link
# against it rather than compiling header-only.

compile_implicit_value_interface <- function() {
  ensure_ode_interface_loaded()

  include_dir <- dirname(dirname(resolve_test_path(
    "include/odelia/ode_solver.hpp", "inst/include/odelia/ode_solver.hpp")))
  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = paste0("-I", shQuote(include_dir)),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <cmath>
    #include <XAD/XAD.hpp>
    #include <odelia/implicit_node.hpp>
    #include <odelia/ode_solver.hpp>

    using tape_type = xad::Tape<double>;
    using adouble = tape_type::active_type;

    // A two-parameter system whose operating point is defined implicitly: the root of
    // a*y^3 + y - b = 0. Written once in the working scalar, so the double root-find
    // and the recorded evaluation read the same equation.
    template <typename T>
    struct CubicBalance {
      using value_type = T;
      T a, b;

      // Declared -> T: a deduced return type here is an XAD expression template
      // holding references to the temporaries of this return statement.
      T residual(T y) const { return a * y * y * y + y - b; }

      // The operating point, in double and off the tape.
      double solve() const {
        double y = 0.0;
        for (int i = 0; i < 100; ++i) {
          const double ad = odelia::util::to_passive(a);
          const double bd = odelia::util::to_passive(b);
          const double f = ad * y * y * y + y - bd;
          const double fp = 3.0 * ad * y * y + 1.0;
          const double dy = f / fp;
          y -= dy;
          if (std::abs(dy) < 1e-15 * (std::abs(y) + 1.0)) break;
        }
        return y;
      }
    };

    // [[Rcpp::export]]
    double implicit_root(double a, double b) {
      CubicBalance<double> sys{a, b};
      return sys.solve();
    }

    // The value and d(y*)/d(a), d(y*)/d(b) from one reverse sweep.
    // [[Rcpp::export]]
    Rcpp::List implicit_value_gradient(double a, double b) {
      tape_type tape;
      CubicBalance<adouble> sys{adouble(a), adouble(b)};
      tape.registerInput(sys.a);
      tape.registerInput(sys.b);
      tape.newRecording();

      const double y_star = sys.solve();
      adouble y = odelia::implicit_value<adouble>(
          y_star, [&](adouble yy) -> adouble { return sys.residual(yy); },
          odelia::denom_sign::positive);

      tape.registerOutput(y);
      xad::derivative(y) = 1.0;
      tape.computeAdjoints();
      return Rcpp::List::create(
          Rcpp::_["value"] = xad::value(y),
          Rcpp::_["y_star"] = y_star,
          Rcpp::_["identical"] = odelia::util::identical(xad::value(y), y_star),
          Rcpp::_["d_da"] = xad::derivative(sys.a),
          Rcpp::_["d_db"] = xad::derivative(sys.b));
    }

    // The same node with the sign of the operating point declared the wrong way
    // round, which is what a fold looks like from the caller.
    // [[Rcpp::export]]
    double implicit_value_wrong_sign(double a, double b) {
      tape_type tape;
      CubicBalance<adouble> sys{adouble(a), adouble(b)};
      tape.registerInput(sys.a);
      tape.registerInput(sys.b);
      tape.newRecording();
      const double y_star = sys.solve();
      adouble y = odelia::implicit_value<adouble>(
          y_star, [&](adouble yy) -> adouble { return sys.residual(yy); },
          odelia::denom_sign::negative);
      return xad::value(y);
    }

    // A one-state system whose rate goes non-finite once the state passes a
    // threshold, so the adaptive controller meets a non-finite error estimate. Its
    // magnitude does not matter: what matters is that std::max drops a NaN, so
    // without the rejection the largest measured error comes from whichever
    // components stayed finite and the step is accepted.
    struct DivergingSystem {
      using value_type = double;
      double y = 1.0, dydt = 1.0, time = 0.0, t0 = 0.0, threshold = 1.5;

      size_t ode_size() const { return 1; }
      double ode_time() const { return time; }
      double ode_t0() const { return t0; }

      template <typename Iterator>
      Iterator set_ode_state(Iterator it, double time_) {
        y = *it++;
        time = time_;
        dydt = (y > threshold) ? std::numeric_limits<double>::quiet_NaN() : 1.0;
        return it;
      }
      template <typename Iterator>
      Iterator set_initial_state(Iterator it, double t0_ = 0.0) {
        t0 = t0_;
        y = *it++;
        return it;
      }
      template <typename Iterator>
      Iterator ode_state(Iterator it) const { *it++ = y; return it; }
      template <typename Iterator>
      Iterator ode_rates(Iterator it) const { *it++ = dydt; return it; }
      void reset() { y = 1.0; time = t0; dydt = 1.0; }
    };

    // Advance the diverging system past its threshold and report the state reached.
    // [[Rcpp::export]]
    std::vector<double> advance_past_non_finite(double t_end) {
      odelia::ode::OdeControl control;
      odelia::ode::Solver<DivergingSystem> solver(DivergingSystem(), control);
      solver.set_state(std::vector<double>{1.0}, 0.0);
      solver.advance_adaptive(std::vector<double>{0.0, t_end});
      return solver.state();
    }', verbose = FALSE)
}

testthat::test_that("implicit_value returns the operating point and its IFT derivative", {
  compile_implicit_value_interface()

  a <- 0.35
  b <- 2.2
  got <- implicit_value_gradient(a, b)

  # The node returns y* itself, so a parameter the equation does not reach can
  # introduce no shift.
  testthat::expect_true(got$identical)

  # Central difference of the same root-find, at three step sizes. The node's dF/dy
  # is itself a central difference at eps = 1e-6*(|y*|+1), so this reports what that
  # costs rather than asserting it away.
  eps <- c(1e-3, 1e-4, 1e-5)
  fd <- vapply(eps, function(h) {
    c((implicit_root(a + h, b) - implicit_root(a - h, b)) / (2 * h),
      (implicit_root(a, b + h) - implicit_root(a, b - h)) / (2 * h))
  }, numeric(2))

  residual <- vapply(seq_along(eps), function(i) {
    max(abs(c(got$d_da, got$d_db) - fd[, i]) / abs(fd[, i]))
  }, numeric(1))
  message(sprintf("IFT vs central difference: eps=%g rel=%.3e | eps=%g rel=%.3e | eps=%g rel=%.3e",
                  eps[[1]], residual[[1]], eps[[2]], residual[[2]],
                  eps[[3]], residual[[3]]))

  # The residual falls as eps^2 across the three, so what it measures is the
  # reference difference's own truncation, not the node's dF/dy: the node's error is
  # below the smallest of these.
  for (i in seq_along(eps)) {
    testthat::expect_lt(residual[[i]], 5 * eps[[i]]^2)
  }
})

testthat::test_that("implicit_value stops when the declared denominator sign is wrong", {
  compile_implicit_value_interface()

  # a > 0 makes dF/dy = 3*a*y^2 + 1 strictly positive, so declaring it negative is
  # the fold case: the node must stop rather than return an inverted gradient.
  testthat::expect_error(implicit_value_wrong_sign(0.35, 2.2), "not invertible")
})

testthat::test_that("a non-finite step-size decision is rejected", {
  compile_implicit_value_interface()

  # The controller shrinks by the largest permitted factor on every attempt and never
  # accepts, so the step loop runs down to the accuracy limit and stops. Accepting
  # instead would commit a non-finite state to the trajectory.
  testthat::expect_error(advance_past_non_finite(2.0),
                         "Cannot achieve the desired accuracy")

  # Below the threshold the rates stay finite and the same run advances normally.
  testthat::expect_equal(advance_past_non_finite(0.25), 1.25, tolerance = 1e-10)
})
