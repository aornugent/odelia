# Tests for OdeElement and the range helpers that walk a container of elements.
# The interface header is header-only, so this needs the include path and no link
# against the odelia shared library.

compile_ode_element_interface <- function() {
  include_dir <- dirname(dirname(resolve_test_path(
    "include/odelia/ode_solver.hpp", "inst/include/odelia/ode_solver.hpp")))
  withr::local_envvar(PKG_CPPFLAGS = paste0("-I", shQuote(include_dir)))
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <odelia/ode_interface.hpp>

    // One state, one rate, one aux, moved through std::vector<value_type> iterators.
    struct Point {
      using value_type = double;
      double x = 0.0;

      std::vector<double>::const_iterator
      set_ode_state(std::vector<double>::const_iterator it) {
        x = *it++;
        return it;
      }
      std::vector<double>::iterator ode_state(std::vector<double>::iterator it) const {
        *it++ = x;
        return it;
      }
      std::vector<double>::iterator ode_rates(std::vector<double>::iterator it) const {
        *it++ = 2.0 * x;
        return it;
      }
      std::vector<double>::iterator ode_aux(std::vector<double>::iterator it) const {
        *it++ = x + 1.0;
        return it;
      }
      std::vector<double>::iterator set_ode_aux(std::vector<double>::iterator it) {
        return ++it;
      }
      size_t ode_size() const { return 1; }
      size_t aux_size() const { return 1; }
    };

    // Moves its state through int iterators rather than the element value_type ones,
    // which is what a double-typed signature does to an element carrying an active
    // scalar.
    struct WrongPoint {
      using value_type = double;
      std::vector<int>::const_iterator
      set_ode_state(std::vector<int>::const_iterator it) { return it; }
      std::vector<int>::iterator ode_state(std::vector<int>::iterator it) const { return it; }
      std::vector<int>::iterator ode_rates(std::vector<int>::iterator it) const { return it; }
      std::vector<int>::iterator ode_aux(std::vector<int>::iterator it) const { return it; }
    };

    static_assert(odelia::ode::OdeElement<Point>);
    static_assert(!odelia::ode::OdeElement<WrongPoint>);

    // [[Rcpp::export]]
    Rcpp::List element_range_round_trip(std::vector<double> state) {
      std::vector<Point> pts(state.size());
      const size_t n = odelia::ode::ode_size(pts.begin(), pts.end());
      odelia::ode::set_ode_state(pts.begin(), pts.end(), state.cbegin());

      std::vector<double> got(n), rates(n),
          aux(odelia::ode::aux_size(pts.begin(), pts.end()));
      auto state_end = odelia::ode::ode_state(pts.begin(), pts.end(), got.begin());
      odelia::ode::ode_rates(pts.begin(), pts.end(), rates.begin());
      auto aux_end = odelia::ode::ode_aux(pts.begin(), pts.end(), aux.begin());
      auto set_aux_end = odelia::ode::set_ode_aux(pts.begin(), pts.end(), aux.begin());

      return Rcpp::List::create(
          Rcpp::_["state"] = got,
          Rcpp::_["rates"] = rates,
          Rcpp::_["aux"] = aux,
          Rcpp::_["state_advance"] = (int)(state_end - got.begin()),
          Rcpp::_["aux_advance"] = (int)(aux_end - aux.begin()),
          Rcpp::_["set_aux_advance"] = (int)(set_aux_end - aux.begin()));
    }', verbose = FALSE)
}

testthat::test_that("the range helpers move each element's state through its own iterator", {
  compile_ode_element_interface()

  s <- c(1.0, 2.0, 3.0)
  r <- element_range_round_trip(s)

  expect_equal(r$state, s)
  expect_equal(r$rates, 2 * s)
  expect_equal(r$aux, s + 1)
})

testthat::test_that("every helper advances the iterator by the size it reports", {
  compile_ode_element_interface()

  s <- c(1.0, 2.0, 3.0, 4.0)
  r <- element_range_round_trip(s)

  expect_equal(r$state_advance, length(s))
  expect_equal(r$aux_advance, length(s))
  expect_equal(r$set_aux_advance, length(s))
})

testthat::test_that("an element typed for the wrong iterator is rejected at the helper", {
  # The two static_asserts above are the assertion: an element whose state does not
  # move through std::vector<value_type>::iterator fails OdeElement rather than the
  # helper's body, and a program that fails a concept does not compile.
  expect_no_error(compile_ode_element_interface())
})
