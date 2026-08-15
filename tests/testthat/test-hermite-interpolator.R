# Tests for odelia::interpolator::hermite_interpolator -- the C1 piecewise cubic
# carrying a value and a slope at each knot.
#
# The interpolant records on a tape when its scalar or its query position is active,
# and the XAD Tape<T,N> template methods are explicitly instantiated only in the
# odelia shared library, so these snippets link against it rather than compiling
# header-only.

compile_hermite_interface <- function() {
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
    #include <limits>
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

    // Laying the lattice and extending it: what a resource field on fixed knots asks
    // for as its domain grows. Returns the nodes each bound produced, and whether the
    // queries the shorter grid already answered moved when it was extended.
    // [[Rcpp::export]]
    Rcpp::List hermite_lattice(double spacing, std::vector<double> bounds,
                               std::vector<double> u, bool start_off_lattice) {
      hermite_interpolator<double> interp;
      if (start_off_lattice) {
        // A grid this class did not lay out, long enough to pass a length test.
        std::vector<double> odd;
        for (int k = 0; k < 40; ++k) odd.push_back(0.5 + 0.37 * k);
        interp.init(odd, field_values(odd), field_slopes(odd));
      }
      std::vector<int> wanted(bounds.size()), sizes(bounds.size());
      std::vector<double> second(bounds.size()), top(bounds.size());
      std::vector<int> held(u.size() * bounds.size(), 1);
      std::vector<double> previous(u.size());
      bool have_previous = false;
      for (std::size_t b = 0; b < bounds.size(); ++b) {
        wanted[b] = static_cast<int>(
            hermite_interpolator<double>::lattice_size(spacing, bounds[b]));
        interp.ensure_lattice(spacing, static_cast<std::size_t>(wanted[b]));
        const std::vector<double> nodes = interp.knots();
        interp.set_data(field_values(nodes), field_slopes(nodes));
        sizes[b] = static_cast<int>(interp.size());
        second[b] = nodes[1];
        top[b] = nodes.back();
        for (std::size_t i = 0; i < u.size(); ++i) {
          const double now = interp.eval(u[i]);
          if (have_previous)
            held[b * u.size() + i] = odelia::util::identical(now, previous[i]);
          previous[i] = now;
        }
        have_previous = true;
      }
      return Rcpp::List::create(Rcpp::_["wanted"] = wanted,
                                Rcpp::_["size"] = sizes,
                                Rcpp::_["second"] = second,
                                Rcpp::_["top"] = top,
                                Rcpp::_["held"] = held);
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
    }

    // What a query outside the knot range reads: the end line, value and slope of
    // the nearest end.
    // [[Rcpp::export]]
    Rcpp::List hermite_outside(std::vector<double> z, double below, double above) {
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
      interp.set_data(field_values(z), field_slopes(z));
      return Rcpp::List::create(
        Rcpp::_["below"] = interp.eval(below),
        Rcpp::_["below_slope"] = interp.slope(below),
        Rcpp::_["above"] = interp.eval(above),
        Rcpp::_["above_slope"] = interp.slope(above),
        Rcpp::_["front_value"] = field.value(z.front()),
        Rcpp::_["front_slope"] = field.slope(z.front()),
        Rcpp::_["back_value"] = field.value(z.back()),
        Rcpp::_["back_slope"] = field.slope(z.back()));
    }

    // A non-finite query compares false against both ends, so it reaches the span
    // lookup. Reading one is not meaningful; landing on a span that exists is,
    // because the alternative is an unspecified index or a read past the last span.
    // [[Rcpp::export]]
    Rcpp::List hermite_nonfinite(std::vector<double> z) {
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
      interp.set_data(field_values(z), field_slopes(z));
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const double inf = std::numeric_limits<double>::infinity();
      double pv, ps;
      interp.value_and_slope(nan, pv, ps);
      return Rcpp::List::create(
        Rcpp::_["nan"] = interp.eval(nan),
        Rcpp::_["nan_slope"] = interp.slope(nan),
        Rcpp::_["nan_pair_value"] = pv,
        Rcpp::_["nan_pair_slope"] = ps,
        Rcpp::_["pos_inf"] = interp.eval(inf),
        Rcpp::_["neg_inf"] = interp.eval(-inf));
    }

    // The residual at the knots themselves, and the error away from them at two
    // refinements of the same lattice.
    // [[Rcpp::export]]
    Rcpp::List hermite_accuracy(int n, std::vector<double> u) {
      std::vector<double> z(n);
      const double spacing = 4.0 / (n - 1);
      for (int k = 0; k < n; ++k) z[k] = k * spacing;
      hermite_interpolator<double> interp;
      interp.set_nodes(z);
      interp.set_data(field_values(z), field_slopes(z));
      std::vector<double> at_knot(z.size()), away(u.size());
      for (std::size_t i = 0; i < z.size(); ++i)
        at_knot[i] = interp.eval(z[i]) - field.value(z[i]);
      for (std::size_t i = 0; i < u.size(); ++i)
        away[i] = std::abs(interp.eval(u[i]) - field.value(u[i]));
      return Rcpp::List::create(Rcpp::_["at_knot"] = at_knot,
                                Rcpp::_["away"] = away);
    }

    // A lattice asked for a spacing other than the one it holds.
    // [[Rcpp::export]]
    Rcpp::List hermite_lattice_respace(double first, double second, double upper) {
      hermite_interpolator<double> interp;
      interp.ensure_lattice(first,
        hermite_interpolator<double>::lattice_size(first, upper));
      const int n1 = static_cast<int>(interp.size());
      const double s1 = interp.knots()[1];
      interp.ensure_lattice(second,
        hermite_interpolator<double>::lattice_size(second, upper));
      return Rcpp::List::create(
        Rcpp::_["first_size"] = n1, Rcpp::_["first_step"] = s1,
        Rcpp::_["second_size"] = static_cast<int>(interp.size()),
        Rcpp::_["second_step"] = interp.knots()[1],
        Rcpp::_["second_top"] = interp.knots().back());
    }

    // [[Rcpp::export]]
    double hermite_lattice_size(double spacing, double upper) {
      return static_cast<double>(
        hermite_interpolator<double>::lattice_size(spacing, upper));
    }

    // [[Rcpp::export]]
    void hermite_ensure_lattice_too_short(double spacing) {
      hermite_interpolator<double> interp;
      interp.ensure_lattice(spacing, 1);
    }', verbose = FALSE)
}

# Three knot sets: uniform (indexed by arithmetic), not (indexed by a search),
# and the shape a resource field builds -- knot k at k * spacing, accumulated by
# multiplication so consecutive gaps are not bit-identical. That last one is what
# the equal-spacing test has to hold on, and it is the only shape a run uses.
knot_sets <- list(uniform = seq(0, 4, length.out = 33),
                  irregular = c(0, 0.07, 0.31, 0.4, 0.95, 1.6, 2.05, 2.9, 3.4, 4),
                  multiplied = (0:40) * 0.1)
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

test_that("the lattice is laid, extended and never moved", {
  compile_hermite_interface()
  spacing <- 0.05
  bounds <- c(0.4, 1.0, 1.0, 4.3, 17.35)
  # Queries strictly inside the first grid, so every later bound must answer them
  # with the same number: an extension adds nodes above, never moves one below.
  u <- c(0.013, 0.11, 0.276, 0.39)
  out <- hermite_lattice(spacing, bounds, u, FALSE)

  # One node clear of the bound, so a query at the bound is inside a span.
  expect_identical(out$wanted, as.integer(ceiling(bounds / spacing) + 2))
  expect_true(all(out$top > bounds))
  # Node k sits at k * spacing, read back off the grid.
  expect_equal(out$second, rep(spacing, length(bounds)))
  expect_equal(out$top, (out$size - 1) * spacing)
  # The grid only grows, and a bound that needs no more nodes changes nothing.
  expect_false(is.unsorted(out$size))
  expect_identical(out$size[[2]], out$size[[3]])
  # Bit-identical below the old top, for every bound after the first.
  expect_true(all(out$held == 1L))
})

testthat::test_that("the interpolant reproduces the data it was built from", {
  compile_hermite_interface()

  # A value and a slope per knot leaves the span polynomial pinned at both ends,
  # so a knot is not approximated: it is returned. Nothing downstream should have
  # to allow for a residual there.
  r <- hermite_accuracy(41, query)
  testthat::expect_true(all(r$at_knot == 0))
})

testthat::test_that("the value converges at the order a cubic span gives", {
  compile_hermite_interface()

  # Away from the knots the error is the span polynomial against the field, which
  # for a smooth field falls as the fourth power of the spacing. This is the
  # accuracy claim that does hold; the SLOPE does not converge on a field whose
  # spans carry curvature breaks, which is a statement about that field and not
  # about this interpolant.
  coarse <- max(hermite_accuracy(41, query)$away)
  fine <- max(hermite_accuracy(321, query)$away)
  order <- log2(coarse / fine) / log2(8)
  message(sprintf("value convergence order over an eightfold refinement: %.2f", order))
  testthat::expect_gt(order, 3.5)
})

testthat::test_that("outside the knots the end line is extended", {
  compile_hermite_interface()

  # A cubic continued past its span runs away; the end value and end slope do not,
  # and they keep the read C1 across the boundary.
  z <- knot_sets$uniform
  below <- -0.5
  above <- 4.5
  r <- hermite_outside(z, below, above)
  testthat::expect_equal(r$below,
                         r$front_value + r$front_slope * (below - z[[1]]))
  testthat::expect_identical(r$below_slope, r$front_slope)
  testthat::expect_equal(r$above,
                         r$back_value + r$back_slope * (above - z[[length(z)]]))
  testthat::expect_identical(r$above_slope, r$back_slope)
})

testthat::test_that("a non-finite query lands on a span that exists", {
  compile_hermite_interface()

  # NaN compares false against both ends, so it reaches the span lookup rather
  # than an early return. Reading one is not meaningful and NaN is the right
  # answer; what matters is that neither route leaves the spans -- the arithmetic
  # one would convert an unspecified index and the search one would run one past
  # the last span. A read that returns is the evidence for both.
  for (z in knot_sets) {
    r <- hermite_nonfinite(z)
    testthat::expect_true(is.nan(r$nan))
    testthat::expect_true(is.nan(r$nan_slope))
    testthat::expect_true(is.nan(r$nan_pair_value))
    testthat::expect_true(is.nan(r$nan_pair_slope))
    # The infinities are outside the range and take the end line, which runs away
    # in the direction the end slope points.
    testthat::expect_true(is.infinite(r$pos_inf))
    testthat::expect_true(is.infinite(r$neg_inf))
  }
})

test_that("a lattice asked for a different spacing is laid again", {
  compile_hermite_interface()

  # Holding a lattice is not the same as holding THIS lattice, and the held grid
  # answers that off its own nodes rather than off a remembered spacing.
  r <- hermite_lattice_respace(0.05, 0.1, 1.0)
  expect_identical(r$first_size, as.integer(ceiling(1.0 / 0.05) + 2))
  expect_equal(r$first_step, 0.05)
  expect_identical(r$second_size, as.integer(ceiling(1.0 / 0.1) + 2))
  expect_equal(r$second_step, 0.1)
  expect_equal(r$second_top, (r$second_size - 1) * 0.1)
})

test_that("a lattice that cannot be laid is refused, and a large one is not", {
  compile_hermite_interface()

  expect_error(hermite_lattice_size(0, 1), "spacing must be positive")
  expect_error(hermite_lattice_size(-0.1, 1), "spacing must be positive")
  expect_error(hermite_lattice_size(0.05, -1), "must not be negative")
  expect_error(hermite_lattice_size(0.05, NaN), "must not be negative")
  expect_error(hermite_ensure_lattice_too_short(0.05), "at least 2 nodes")

  # lattice_size is arithmetic and does not allocate, so a count no memory could
  # hold is a number rather than an error. Whoever asked knows what ran away and
  # refuses in its own words; this class does not.
  expect_equal(hermite_lattice_size(1e-9, 1e9), 1e18)
})

test_that("a grid the lattice did not lay out is replaced, not extended", {
  compile_hermite_interface()
  spacing <- 0.05
  # The off-lattice grid is 40 nodes reaching 15.0, so it is long enough for the
  # first bound and sits nowhere near k * spacing. Length alone would keep it.
  out <- hermite_lattice(spacing, c(0.4), numeric(0), TRUE)
  expect_identical(out$size[[1]], as.integer(ceiling(0.4 / spacing) + 2))
  expect_equal(out$second[[1]], spacing)
  expect_equal(out$top[[1]], (out$size[[1]] - 1) * spacing)
})
