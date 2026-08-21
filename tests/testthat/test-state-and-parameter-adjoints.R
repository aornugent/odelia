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

    static int tiny_rebinds = 0;

    // Two parameters, mixed into two outputs so that a swapped half, a dropped
    // offset or a transposed sweep gives different numbers rather than the same
    // ones in a different order.
    //
    //   y0 = a*x0 + b*x1      y1 = x0*x1 + square
    //
    // `square` is DERIVED -- written from an expression rather than handed a
    // fresh value -- which is the case a copy between recordings gets wrong:
    // assigning from an expression keeps the slot the target already had.
    template <typename T>
    struct TinySystem {
      using value_type = T;
      T a{T(2.0)}, b{T(-3.0)}, square{T(4.0)};

      std::vector<T*> ad_parameters() { return {&a, &b}; }
      void refresh() { square = a * a; }

      template <typename> friend struct TinySystem;

      template <class U>
      TinySystem<U> rebind_from() const {
        ++tiny_rebinds;
        TinySystem<U> out;
        out.a = U(odelia::util::to_passive(a));
        out.b = U(odelia::util::to_passive(b));
        out.refresh();
        return out;
      }
    };

    // `n_calls` products onto one accumulator, so the accumulate half is visible:
    // the state adjoints are whatever the last call wrote, the parameter adjoints
    // are the sum over calls. `width` sets how many times the evaluation chains
    // through the System, so consecutive recordings can be given different shapes.
    // [[Rcpp::export]]
    Rcpp::List sap_products(std::vector<double> state,
                            std::vector<double> prior, int n_calls, int width) {
      const odelia::ode::row_batch seeds = odelia::ode::row_batch::all_rows(2);
      odelia::ode::row_batch state_adjoint;
      odelia::ode::row_batch parameter_adjoint(2, prior.size());
      for (std::size_t m = 0; m < 2; ++m) {
        std::copy(prior.begin(), prior.end(), parameter_adjoint[m].begin());
      }

      double recording = 0.0;
      xad::adj<double>::tape_type tape(false);
      TinySystem<double> system;
      tiny_rebinds = 0;
      for (int k = 0; k < n_calls; ++k) {
        auto evaluate = [&](TinySystem<adouble>& sys,
                            std::vector<adouble>::const_iterator x,
                            std::vector<adouble>& y) -> void {
          for (int w = 0; w < width; ++w) {
            sys.refresh();
          }
          y[0] = sys.a * x[0] + sys.b * x[1];
          y[1] = x[0] * x[1] + sys.square;
        };
        recording = double(odelia::ode::state_and_parameter_adjoints(
            tape, system, state, seeds, evaluate, state_adjoint,
            parameter_adjoint));
      }
      return Rcpp::List::create(
          Rcpp::_["state"] = Rcpp::wrap(state_adjoint.to_rows()),
          Rcpp::_["parameter"] = Rcpp::wrap(parameter_adjoint.to_rows()),
          Rcpp::_["rebinds"] = tiny_rebinds,
          Rcpp::_["recording"] = recording);
    }

    // Two recordings of DIFFERENT shape in sequence, on one tape: the second
    // returns the parameter rows of the first if the System it records on is
    // fresh, and rows that are wrong but finite if it is not.
    // [[Rcpp::export]]
    Rcpp::List sap_after_wider(std::vector<double> state, int first_width) {
      const odelia::ode::row_batch seeds = odelia::ode::row_batch::all_rows(2);
      odelia::ode::row_batch state_adjoint;
      odelia::ode::row_batch parameter_adjoint(2, 2);
      xad::adj<double>::tape_type tape(false);
      TinySystem<double> system;

      for (int k = 0; k < 2; ++k) {
        const int width = (k == 0) ? first_width : 1;
        auto evaluate = [&](TinySystem<adouble>& sys,
                            std::vector<adouble>::const_iterator x,
                            std::vector<adouble>& y) -> void {
          for (int w = 0; w < width; ++w) {
            sys.refresh();
          }
          y[0] = sys.a * x[0] + sys.b * x[1];
          y[1] = x[0] * x[1] + sys.square;
        };
        if (k == 1) {
          parameter_adjoint.assign(2, 2);
        }
        odelia::ode::state_and_parameter_adjoints(tape, system, state, seeds,
                                                  evaluate, state_adjoint,
                                                  parameter_adjoint);
      }
      return Rcpp::List::create(Rcpp::_["state"] = Rcpp::wrap(state_adjoint.to_rows()),
                                Rcpp::_["parameter"] = Rcpp::wrap(parameter_adjoint.to_rows()));
    }

    // The two shapes it still refuses, and the one it no longer can: 0 gives one
    // accumulator row against two seeds, 1 gives a batch of the wrong width, and
    // 2 hands the same batch in for both halves. A batch whose rows disagree with
    // each other is not among them -- one width covers every row, so that shape
    // cannot be built.
    // [[Rcpp::export]]
    void sap_refused(int which) {
      TinySystem<double> system;
      std::vector<double> state{5.0, 7.0};
      const odelia::ode::row_batch seeds = odelia::ode::row_batch::all_rows(2);
      odelia::ode::row_batch state_adjoint;
      odelia::ode::row_batch parameter_adjoint;
      if (which == 0) {
        parameter_adjoint.assign(1, 2);
      } else if (which == 1) {
        parameter_adjoint.assign(2, 3);
      } else {
        parameter_adjoint.assign(2, 2);
      }

      auto evaluate = [&](TinySystem<adouble>& sys,
                          std::vector<adouble>::const_iterator x,
                          std::vector<adouble>& y) -> void {
        y[0] = sys.a * x[0] + sys.b * x[1];
        y[1] = x[0] * x[1] + sys.square;
      };

      xad::adj<double>::tape_type tape(false);
      odelia::ode::row_batch& out =
          (which == 2) ? parameter_adjoint : state_adjoint;
      odelia::ode::state_and_parameter_adjoints(tape, system, state, seeds,
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
  got <- sap_products(c(5, 7), c(0, 0), 1L, 1L)
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
  once <- sap_products(c(5, 7), c(0, 0), 1L, 1L)
  thrice <- sap_products(c(5, 7), c(0, 0), 3L, 1L)

  # A parameter is reached once per step and its gradient is the sum over the
  # steps swept, so three products of the same block is three times one.
  testthat::expect_equal(rows(thrice$parameter), 3 * rows(once$parameter))

  # The state adjoints are the last sweep's, not a running sum.
  testthat::expect_equal(rows(thrice$state), rows(once$state))

  # And what the caller had in the accumulator is added to, not replaced.
  seeded <- sap_products(c(5, 7), c(10, 20), 1L, 1L)
  testthat::expect_equal(rows(seeded$parameter),
                         rows(once$parameter) +
                           matrix(c(10, 10, 20, 20), nrow = 2))
})

testthat::test_that("the System is lifted once per recording, so no recording inherits another's slots", {
  compile_sap_interface()

  rows <- function(x) do.call(rbind, x)

  # One lift per call, taken here rather than by the caller. A caller holding one
  # System across calls is what this replaced, and it cannot be expressed: the
  # copy is made where the requirement is.
  testthat::expect_identical(sap_products(c(5, 7), c(0, 0), 1L, 1L)$rebinds, 1L)
  testthat::expect_identical(sap_products(c(5, 7), c(0, 0), 3L, 1L)$rebinds, 3L)

  # A recording that follows a WIDER one has to give the same rows as one that
  # follows nothing. This is the case a carried System gets wrong: clearing the
  # tape returns its slot counter to zero, and a derived scalar still holding the
  # wider recording's slot is handed the same number as a live variable in this
  # one. The rows stay finite and plausible; only the numbers move.
  after_wide <- rows(sap_after_wider(c(5, 7), 4L)$parameter)
  alone <- rows(sap_products(c(5, 7), c(0, 0), 1L, 1L)$parameter)
  testthat::expect_equal(after_wide, alone)
})

testthat::test_that("state_and_parameter_adjoints refuses the shapes it cannot honour", {
  compile_sap_interface()

  # One accumulator row against two seeds: the second seed would have nowhere to
  # add to.
  testthat::expect_error(sap_refused(0L), "one row of parameter adjoints")

  # A batch that is not one entry per parameter. Unchecked, this reads and writes
  # past each row for every parameter past its end. Asked of the batch, because
  # one width covers every row -- a batch whose rows disagree with each other is
  # not a shape that can be built, so there is nothing to test per row.
  testthat::expect_error(sap_refused(1L), "[Ii]ncorrect length")

  # The same batch for both halves: the state half reshapes what the parameter
  # half has already been checked for and is about to accumulate into.
  testthat::expect_error(sap_refused(2L), "cannot be the same batch")
})
