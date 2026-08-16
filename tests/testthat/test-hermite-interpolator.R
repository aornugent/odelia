# Tests for odelia::interpolator::hermite_interpolator -- the C1 piecewise cubic
# carrying a value and a slope at each knot.
#
# The interpolant records on a tape when its scalar or its query position is active,
# and the XAD Tape<T,N> template methods are explicitly instantiated only in the
# odelia shared library, so these snippets link against it rather than compiling
# header-only.

compile_hermite_interface <- function() {
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
    #include <odelia/hermite_interpolator.hpp>

    using tape_type = xad::Tape<double>;
    using adouble = tape_type::active_type;
    using odelia::interpolator::hermite_interpolator;

    // A vertical field sampled at knots, with the slope from the same expression the
    // value comes from -- the pair a crown integral reads.
    struct AttenuatingField {
      double extinction = 1.7, amplitude = 2.3;

      double value(double z) const { return amplitude * std::exp(-extinction * z); }
      double slope(double z) const { return -extinction * value(z); }
    };

    static AttenuatingField field;

    static std::vector<double> field_values(const std::vector<double>& z) {
      std::vector<double> v(z.size());
      for (std::size_t i = 0; i < z.size(); ++i) v[i] = field.value(z[i]);
      return v;
    }
    static std::vector<double> field_slopes(const std::vector<double>& z) {
      std::vector<double> v(z.size());
      for (std::size_t i = 0; i < z.size(); ++i) v[i] = field.slope(z[i]);
      return v;
    }

    // [[Rcpp::export]]
    Rcpp::List hermite_value_and_slope(std::vector<double> z, std::vector<double> u) {
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
      interp.set_data(field_values(z), field_slopes(z));
      std::vector<double> value(u.size()), slope(u.size()),
                          pair_value(u.size()), pair_slope(u.size());
      for (std::size_t i = 0; i < u.size(); ++i) {
        value[i] = interp.eval(u[i]);
        slope[i] = interp.slope(u[i]);
        interp.value_and_slope(u[i], pair_value[i], pair_slope[i]);
      }
      return Rcpp::List::create(Rcpp::_["value"] = value, Rcpp::_["slope"] = slope,
                                Rcpp::_["pair_value"] = pair_value,
                                Rcpp::_["pair_slope"] = pair_slope);
    }

    // Setting the nodes once and the data twice, against two all-at-once builds on the
    // same nodes: what a per-stage rebuild does, compared with rebuilding everything.
    // [[Rcpp::export]]
    Rcpp::List hermite_split_matches_all_at_once(std::vector<double> z,
                                                 std::vector<double> u,
                                                 double scale) {
      const std::vector<double> y = field_values(z), m = field_slopes(z);
      std::vector<double> y2(y.size()), m2(m.size());
      for (std::size_t i = 0; i < y.size(); ++i) { y2[i] = scale * y[i]; m2[i] = scale * m[i]; }

      hermite_interpolator<double> split, whole;
      split.set_nodes(z);

      std::vector<int> same(2 * u.size());
      split.set_data(y, m);
      whole.init(z, y, m);
      for (std::size_t i = 0; i < u.size(); ++i) {
        same[i] = odelia::util::identical(split.eval(u[i]), whole.eval(u[i])) &&
                  odelia::util::identical(split.slope(u[i]), whole.slope(u[i]));
      }
      // The second set_data reuses the span layout the first one wrote into.
      split.set_data(y2, m2);
      whole.init(z, y2, m2);
      for (std::size_t i = 0; i < u.size(); ++i) {
        same[u.size() + i] =
            odelia::util::identical(split.eval(u[i]), whole.eval(u[i])) &&
            odelia::util::identical(split.slope(u[i]), whole.slope(u[i]));
      }
      return Rcpp::List::create(Rcpp::_["identical"] = same,
                                Rcpp::_["initialised"] = split.is_initialised(),
                                Rcpp::_["size"] = static_cast<int>(split.size()));
    }

    // d(value)/d(u) recorded by a read at an active position.
    // [[Rcpp::export]]
    std::vector<double> hermite_active_position_adjoint(std::vector<double> z,
                                                        std::vector<double> u) {
      const std::vector<double> yd = field_values(z), md = field_slopes(z);
      const std::vector<adouble> y(yd.begin(), yd.end()), m(md.begin(), md.end());
      hermite_interpolator<adouble> interp;
      interp.set_nodes(z);
      interp.set_data(y, m);

      std::vector<double> d(u.size());
      for (std::size_t i = 0; i < u.size(); ++i) {
        tape_type tape;
        adouble ui = u[i];
        tape.registerInput(ui);
        tape.newRecording();
        adouble value = interp.eval(ui);
        tape.registerOutput(value);
        xad::derivative(value) = 1.0;
        tape.computeAdjoints();
        d[i] = xad::derivative(ui);
      }
      return d;
    }

    // [[Rcpp::export]]
    double hermite_eval_before_data(std::vector<double> z) {
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
      return interp.eval(z.front());
    }

    // [[Rcpp::export]]
    void hermite_data_before_nodes(std::vector<double> z) {
      hermite_interpolator<double> interp;
      interp.set_data(field_values(z), field_slopes(z));
    }

    // [[Rcpp::export]]
    void hermite_set_nodes_descending(std::vector<double> z) {
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
    }', verbose = FALSE)
}

# Two knot sets: uniform (indexed by arithmetic) and not (indexed by a search).
knot_sets <- list(uniform = seq(0, 4, length.out = 33),
                  irregular = c(0, 0.07, 0.31, 0.4, 0.95, 1.6, 2.05, 2.9, 3.4, 4))
query <- c(0.05, 0.31, 0.62, 1.0, 1.55, 2.4, 3.87, 3.999)

testthat::test_that("slope is the exact derivative of what eval returns", {
  compile_hermite_interface()

  # The slope is not a fit by-product: it is the derivative of the same cubic eval
  # uses, so it agrees with a central difference of eval everywhere, knots included.
  h <- 1e-6
  for (z in knot_sets) {
    r <- hermite_value_and_slope(z, query)
    fd <- (hermite_value_and_slope(z, query + h)$value -
           hermite_value_and_slope(z, query - h)$value) / (2 * h)
    testthat::expect_equal(r$slope, fd, tolerance = 1e-8)

    # value_and_slope loads the span once; it must return the same pair.
    testthat::expect_identical(r$pair_value, r$value)
    testthat::expect_identical(r$pair_slope, r$slope)
  }
})

testthat::test_that("setting the nodes once and the data per stage moves no value", {
  compile_hermite_interface()

  for (z in knot_sets) {
    r <- hermite_split_matches_all_at_once(z, query, scale = 0.37)
    testthat::expect_true(all(r$identical == 1L))
    testthat::expect_true(r$initialised)
    testthat::expect_equal(r$size, length(z))
  }
})

testthat::test_that("a read at an active position records the derivative of the query", {
  compile_hermite_interface()

  # Without the graft the span is indexed at the passive position and nothing records
  # the query, so this adjoint would be exactly zero.
  h <- 1e-6
  for (z in knot_sets) {
    adjoint <- hermite_active_position_adjoint(z, query)
    testthat::expect_false(all(adjoint == 0))

    fd <- (hermite_value_and_slope(z, query + h)$value -
           hermite_value_and_slope(z, query - h)$value) / (2 * h)
    worst <- max(abs(adjoint - fd) / abs(fd))
    message(sprintf("active-position adjoint vs central difference (h=%g): worst rel=%.3e",
                    h, worst))
    testthat::expect_lt(worst, 1e-7)
  }
})

testthat::test_that("an incomplete or unusable build stops", {
  compile_hermite_interface()

  z <- knot_sets$uniform
  testthat::expect_error(hermite_eval_before_data(z), "not initialised")
  testthat::expect_error(hermite_data_before_nodes(z), "no knots set")
  testthat::expect_error(hermite_set_nodes_descending(rev(z)), "strictly ascending")
  testthat::expect_error(hermite_set_nodes_descending(c(1.0)), "at least 2 knots")
})
