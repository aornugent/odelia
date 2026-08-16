# Tests for the two ways a value computed away from the tape gets onto it:
# record_with_derivatives, which takes the derivatives as supplied numbers, and
# implicit_value, which derives them from a residual through the implicit
# function theorem and records the result through the first.
#
# Both record on a tape, and the XAD Tape<T,N> template methods are
# explicitly instantiated only in the odelia shared library, so these snippets link
# against it rather than compiling header-only.

compile_implicit_value_interface <- function() {
  ensure_ode_interface_loaded()

  include_dir <- odelia_include_dir()
  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(include_dir),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <cmath>
    #include <XAD/XAD.hpp>
    #include <odelia/implicit_node.hpp>

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
          y_star, [&](adouble yy) -> adouble { return sys.residual(yy); });

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

    // A value the tape never saw computed, entering it against two inputs it
    // never touched. Nothing here relates `value` to u or v, so the derivatives
    // that come back can only be the supplied ones.
    // [[Rcpp::export]]
    Rcpp::List record_with_derivatives_gradient(double value, double du,
                                                double dv) {
      tape_type tape;
      adouble u(1.5), v(-0.25);
      tape.registerInput(u);
      tape.registerInput(v);
      tape.newRecording();

      adouble y = odelia::record_with_derivatives<adouble>(value,
                                                           {{u, du}, {v, dv}});
      tape.registerOutput(y);
      xad::derivative(y) = 1.0;
      tape.computeAdjoints();
      return Rcpp::List::create(
          Rcpp::_["value"] = xad::value(y),
          Rcpp::_["identical"] = odelia::util::identical(xad::value(y), value),
          Rcpp::_["d_du"] = xad::derivative(u),
          Rcpp::_["d_dv"] = xad::derivative(v));
    }

    // A residual that touches zero rather than crossing it, so dF/dy is zero at
    // the operating point and the quotient the theorem asks for does not exist.
    // [[Rcpp::export]]
    double implicit_value_at_fold(double a) {
      tape_type tape;
      adouble aa(a);
      tape.registerInput(aa);
      tape.newRecording();
      adouble y = odelia::implicit_value<adouble>(
          a, [&](adouble yy) -> adouble { return (yy - aa) * (yy - aa); });
      return xad::value(y);
    }', verbose = FALSE)
}

testthat::test_that("record_with_derivatives returns the value and carries what it was given", {
  compile_implicit_value_interface()

  got <- record_with_derivatives_gradient(4.75, 2.0, -3.5)

  # Each term is an input minus its own passive copy, which is zero in value, so
  # what comes back is the number handed in rather than one near it.
  testthat::expect_true(got$identical)
  testthat::expect_equal(got$d_du, 2.0)
  testthat::expect_equal(got$d_dv, -3.5)

  # A non-finite derivative poisons the VALUE and not only the tape, because NaN
  # times zero is not a number. It is refused here, where the two are still
  # separable; downstream they are one expression.
  testthat::expect_error(record_with_derivatives_gradient(4.75, NaN, 1.0),
                         "is not finite")
})

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

testthat::test_that("implicit_value stops where the theorem does not apply", {
  compile_implicit_value_interface()

  # F(y) = (y - a)^2 is zero at y* = a and so is its slope there, so the node must
  # stop rather than divide by it and return a gradient nothing supports.
  testthat::expect_error(implicit_value_at_fold(0.35), "does not apply")
})

