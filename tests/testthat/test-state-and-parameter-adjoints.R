# Tests for state_and_parameter_adjoints -- one recording over a System's state
# followed by its parameters, swept once per seed and split back along the same
# seam. The two halves are treated differently on purpose: the state adjoints are
# replaced and the parameter adjoints accumulated.
#
# It records on a tape, and the XAD Tape<T,N> template methods are explicitly
# instantiated only in the odelia shared library, so this snippet links against it
# rather than compiling header-only.

compile_sap_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <XAD/XAD.hpp>
    #include <odelia/gradient.hpp>

    using adouble = odelia::ode::active_scalar<double>;

    // Two states and two parameters, mixed into two outputs so that a swapped
    // half, a dropped offset or a transposed sweep gives different numbers rather
    // than the same ones in a different order.
    //
    //   y0 = a*x0 + b*x1      y1 = x0*x1 + a*a
    struct TinySystem {
      adouble a{2.0}, b{-3.0};
      std::vector<adouble*> ad_parameters() { return {&a, &b}; }
    };

    // `n_calls` products onto one accumulator, so the accumulate half is visible:
    // the state adjoints are whatever the last call wrote, the parameter adjoints
    // are the sum over calls.
    // [[Rcpp::export]]
    Rcpp::List sap_products(std::vector<double> state,
                            std::vector<double> prior, int n_calls,
                            bool fresh_copy) {
      std::vector<std::vector<double>> seeds{{1.0, 0.0}, {0.0, 1.0}};
      std::vector<std::vector<double>> state_adjoint;
      std::vector<std::vector<double>> parameter_adjoint(2, prior);

      double recording = 0.0;
      xad::adj<double>::tape_type tape(false);
      TinySystem kept;
      for (int k = 0; k < n_calls; ++k) {
        TinySystem made;
        TinySystem& sys = fresh_copy ? made : kept;
        auto evaluate = [&](std::vector<adouble>::const_iterator x,
                            std::vector<adouble>& y) -> void {
          y[0] = sys.a * x[0] + sys.b * x[1];
          y[1] = x[0] * x[1] + sys.a * sys.a;
        };
        recording = double(odelia::ode::state_and_parameter_adjoints(
            tape, sys, state, seeds, evaluate, state_adjoint,
            parameter_adjoint));
      }
      return Rcpp::List::create(
          Rcpp::_["state"] = Rcpp::wrap(state_adjoint),
          Rcpp::_["parameter"] = Rcpp::wrap(parameter_adjoint),
          Rcpp::_["recording"] = recording);
    }

    // The three shapes it refuses, selected so one export covers them: 0 gives
    // one accumulator row against two seeds, 1 gives a row of the wrong width,
    // 2 hands the same vector in for both halves.
    // [[Rcpp::export]]
    void sap_refused(int which) {
      TinySystem sys;
      std::vector<double> state{5.0, 7.0};
      std::vector<std::vector<double>> seeds{{1.0, 0.0}, {0.0, 1.0}};
      std::vector<std::vector<double>> state_adjoint;
      std::vector<std::vector<double>> parameter_adjoint;
      if (which == 0) {
        parameter_adjoint.assign(1, std::vector<double>(2, 0.0));
      } else if (which == 1) {
        parameter_adjoint.assign(2, std::vector<double>(3, 0.0));
      } else {
        parameter_adjoint.assign(2, std::vector<double>(2, 0.0));
      }

      auto evaluate = [&](std::vector<adouble>::const_iterator x,
                          std::vector<adouble>& y) -> void {
        y[0] = sys.a * x[0] + sys.b * x[1];
        y[1] = x[0] * x[1] + sys.a * sys.a;
      };

      xad::adj<double>::tape_type tape(false);
      std::vector<std::vector<double>>& out =
          (which == 2) ? parameter_adjoint : state_adjoint;
      odelia::ode::state_and_parameter_adjoints(tape, sys, state, seeds,
                                                evaluate, out,
                                                parameter_adjoint);
    }', verbose = FALSE)
}

testthat::test_that("state_and_parameter_adjoints splits the sweep along the seam it packed", {
  compile_sap_interface()

  # a = 2, b = -3, x = (5, 7). By hand, from
  #   y0 = a*x0 + b*x1        y1 = x0*x1 + a*a
  # d(y0)/d(x) = (a, b) = (2, -3)     d(y0)/d(p) = (x0, x1) = (5, 7)
  # d(y1)/d(x) = (x1, x0) = (7, 5)    d(y1)/d(p) = (2a, 0)  = (4, 0)
  got <- sap_products(c(5, 7), c(0, 0), 1L, TRUE)
  state <- do.call(rbind, got$state)
  parameter <- do.call(rbind, got$parameter)

  testthat::expect_equal(state[1, ], c(2, -3))
  testthat::expect_equal(state[2, ], c(7, 5))
  testthat::expect_equal(parameter[1, ], c(5, 7))
  testthat::expect_equal(parameter[2, ], c(4, 0))

  # Six distinct values over eight entries, so a transposed sweep or a half read
  # at the wrong offset cannot pass by coincidence.
  testthat::expect_equal(length(unique(c(state, parameter))), 6L)
  testthat::expect_gt(got$recording, 0)
})

testthat::test_that("the parameter half accumulates and the state half does not", {
  compile_sap_interface()

  rows <- function(x) do.call(rbind, x)
  once <- sap_products(c(5, 7), c(0, 0), 1L, TRUE)
  thrice <- sap_products(c(5, 7), c(0, 0), 3L, TRUE)

  # A parameter is reached once per step and its gradient is the sum over the
  # steps swept, so three products of the same block is three times one.
  testthat::expect_equal(rows(thrice$parameter), 3 * rows(once$parameter))

  # The state adjoints are the last sweep's, not a running sum.
  testthat::expect_equal(rows(thrice$state), rows(once$state))

  # And what the caller had in the accumulator is added to, not replaced.
  seeded <- sap_products(c(5, 7), c(10, 20), 1L, TRUE)
  testthat::expect_equal(rows(seeded$parameter),
                         rows(once$parameter) +
                           matrix(c(10, 10, 20, 20), nrow = 2))
})

testthat::test_that("the active System has to be rebound per call, and that is not advisory", {
  compile_sap_interface()

  rows <- function(x) do.call(rbind, x)
  fresh <- sap_products(c(5, 7), c(0, 0), 3L, TRUE)
  kept <- sap_products(c(5, 7), c(0, 0), 3L, FALSE)

  # A active System carried across calls has its parameters written from inputs of a
  # recording that has since been cleared. Nothing detects it, so this pins that
  # the requirement is load-bearing rather than a precaution: were reuse to become
  # safe, this fails and the contract above it is what needs rewriting.
  testthat::expect_false(isTRUE(all.equal(rows(kept$parameter),
                                          rows(fresh$parameter))))

  # And the failure is partial, which is the reason it needs a test at all: the
  # first seed's rows are exact in both.
  testthat::expect_equal(rows(kept$parameter)[1, ], rows(fresh$parameter)[1, ])
  testthat::expect_false(isTRUE(all.equal(rows(kept$parameter)[2, ],
                                          rows(fresh$parameter)[2, ])))
})

testthat::test_that("state_and_parameter_adjoints refuses the shapes it cannot honour", {
  compile_sap_interface()

  # One accumulator row against two seeds: the second seed would have nowhere to
  # add to.
  testthat::expect_error(sap_refused(0L), "one row of parameter adjoints")

  # A row that is not one entry per parameter. Unchecked, this reads and writes
  # past the row for every parameter past its end.
  testthat::expect_error(sap_refused(1L), "[Ii]ncorrect length")

  # The same vector for both halves: the state half resizes what the parameter
  # half has already been checked for and is about to accumulate into.
  testthat::expect_error(sap_refused(2L), "cannot be the same vector")
})
