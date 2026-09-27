# Tests that the solver records the step size each accepted step took, that a
# replay driven by those sizes reproduces the adaptive run bitwise where a replay
# driven by the recorded times does not, and that a walk over a recording stores
# its own rows.

source_odelia_cpp <- function(code) {
  ensure_ode_interface_loaded()

  include_dir <- odelia_include_dir()
  odelia_so <- .odelia_test_cache$odelia_so
  pkg_libs <- if (is.character(odelia_so) &&
                  length(odelia_so) == 1 &&
                  !is.na(odelia_so) &&
                  nzchar(odelia_so) &&
                  file.exists(odelia_so)) {
    shQuote(normalizePath(odelia_so, winslash = "/", mustWork = FALSE))
  } else {
    Sys.getenv("PKG_LIBS", unset = "")
  }
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(include_dir),
    PKG_LIBS = pkg_libs
  )
  Rcpp::sourceCpp(code = code, verbose = FALSE)
}

compile_step_record_interface <- function() {
  source_odelia_cpp('
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <vector>
    #include <odelia/ode_solver.hpp>
    #include <examples/lorenz_system.hpp>

    using namespace odelia;
    using LorenzD = LorenzSystem<double>;

    static ode::Solver<LorenzD> make_solver(Rcpp::NumericVector y0, double t0) {
      LorenzD sys(10.0, 28.0, 8.0 / 3.0);
      std::vector<double> y(y0.begin(), y0.end());
      sys.set_initial_state(y.begin(), t0);
      sys.reset();
      ode::Solver<LorenzD> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      return solver;
    }

    // Adaptive run over [t0, t1], then the same trajectory replayed twice: once
    // over the recorded step sizes, once over the recorded times.
    // [[Rcpp::export]]
    Rcpp::List step_record_replays(Rcpp::NumericVector y0, double t0, double t1) {
      ode::Solver<LorenzD> solver = make_solver(y0, t0);
      solver.advance_adaptive({t0, t1});
      const std::vector<double> times = solver.times();
      const std::vector<double> step_sizes = solver.step_sizes();
      const std::vector<double> y_adaptive = solver.state();

      std::vector<ode::instruction> recording;
      recording.reserve(times.size());
      for (size_t i = 0; i < times.size(); ++i) {
        recording.push_back({times[i], step_sizes[i]});
      }
      ode::Solver<LorenzD> by_steps = make_solver(y0, t0);
      by_steps.advance_recorded(recording);

      ode::Solver<LorenzD> by_times = make_solver(y0, t0);
      by_times.advance_fixed(times);

      return Rcpp::List::create(
        Rcpp::Named("times") = Rcpp::wrap(times),
        Rcpp::Named("step_sizes") = Rcpp::wrap(step_sizes),
        Rcpp::Named("y_adaptive") = Rcpp::wrap(y_adaptive),
        Rcpp::Named("y_by_steps") = Rcpp::wrap(by_steps.state()),
        Rcpp::Named("y_by_times") = Rcpp::wrap(by_times.state()),
        Rcpp::Named("time_by_steps") = by_steps.time(),
        Rcpp::Named("time_adaptive") = solver.time());
    }
  ')
}

bits <- function(x) vapply(x, function(v) paste(writeBin(v, raw()), collapse = ""),
                           character(1))

test_that("the solver records a step size beside each recorded time", {
  compile_step_record_interface()
  # t of order 100 against h of order 0.01, where fl(fl(t + h) - t) != h bites.
  res <- step_record_replays(c(1, 1, 1), 100, 101)

  expect_equal(length(res$step_sizes), length(res$times))
  expect_true(is.nan(res$step_sizes[1]))
  expect_true(all(is.finite(res$step_sizes[-1])))
  expect_true(all(res$step_sizes[-1] > 0))

  # sum(h) reaches the final time only approximately; that is why h is recorded.
  reached <- res$times[1] + sum(res$step_sizes[-1])
  expect_equal(reached, 101, tolerance = 1e-12)
  cat(sprintf("steps=%d h range=[%.6g, %.6g] t0+sum(h)-t1=%.3g\n",
              length(res$step_sizes) - 1L, min(res$step_sizes[-1]),
              max(res$step_sizes[-1]), reached - 101))
})

test_that("a replay over the recorded step sizes reproduces the adaptive run bitwise", {
  compile_step_record_interface()
  res <- step_record_replays(c(1, 1, 1), 100, 101)

  expect_identical(bits(res$y_by_steps), bits(res$y_adaptive))

  # And the time-driven replay does not: differencing the recorded times cannot
  # recover the step sizes the adaptive run took.
  differing <- sum(bits(res$y_by_times) != bits(res$y_adaptive))
  worst <- max(abs(res$y_by_times - res$y_adaptive))
  cat(sprintf("time-driven replay: %d of %d components differ, worst |diff|=%.6g\n",
              differing, length(res$y_adaptive), worst))
  expect_gt(differing, 0)
})


# A decay whose state gains a component at an insertion, and each of whose rate
# evaluations stores the time it ran at: the smallest System that both solves for
# values and widens.
compile_widening_interface <- function() {
  source_odelia_cpp('
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <limits>
    #include <span>
    #include <vector>
    #include <odelia/ode_solver.hpp>

    using namespace odelia;

    struct Widening {
      using value_type = double;
      struct solved_values {
        double seeded = std::numeric_limits<double>::quiet_NaN();
        double found = std::numeric_limits<double>::quiet_NaN();
      };

      size_t ode_size() const { return y.size(); }
      double ode_time() const { return time; }
      void reset() { y.assign(1, 1.0); time = 0.0; }
      template <typename It> It set_ode_state(It it, double t) {
        for (double& v : y) v = *it++;
        time = t;
        return it;
      }
      template <typename It> It ode_state(It it) const {
        for (double v : y) *it++ = v;
        return it;
      }
      template <typename It> It ode_rates(It it) {
        if (slot != nullptr) slot->found = time;
        for (size_t i = 0; i < y.size(); ++i) *it++ = -(1.0 + i) * y[i];
        return it;
      }

      // What the slot held when this evaluation began, and then its own value.
      void store_solved(solved_values& into) {
        into.seeded = into.found;
        slot = &into;
      }
      void load_solved(const solved_values&) {}
      void end_solved() { slot = nullptr; }

      template <class It>
      void apply_insertion(double t, It x, std::vector<double>& out) {
        for (double& v : y) v = *x++;
        time = t;
        y.push_back(1.0);
        out = y;
      }
      void set_recorded_state(const std::vector<double>& s, double t) {
        y = s;
        time = t;
      }

      std::vector<double> y{1.0};
      double time = 0.0;
      solved_values* slot = nullptr;
    };

    using Record = ode::step_record<Widening>;

    static ode::Solver<Widening> kept_solver() {
      ode::Solver<Widening> solver{Widening{}, ode::OdeControl{}};
      solver.set_collect(false);
      solver.set_keep_states(true);
      solver.reset();
      return solver;
    }

    static Rcpp::List describe(std::span<const Record> rec) {
      Rcpp::LogicalVector insertion;
      Rcpp::NumericVector time, size, seeded, found;
      Rcpp::IntegerVector width;
      for (const Record& row : rec) {
        insertion.push_back(row.insertion);
        time.push_back(row.time);
        size.push_back(row.step_size);
        width.push_back(static_cast<int>(row.state.size()));
        for (const Widening::solved_values& v : row.solved) {
          seeded.push_back(v.seeded);
          found.push_back(v.found);
        }
      }
      return Rcpp::List::create(
        Rcpp::Named("insertion") = insertion, Rcpp::Named("time") = time,
        Rcpp::Named("size") = size, Rcpp::Named("width") = width,
        Rcpp::Named("seeded") = seeded, Rcpp::Named("found") = found,
        Rcpp::Named("state_end") = Rcpp::wrap(rec.back().state));
    }

    // Two adaptive intervals with an insertion between them, then a walk over
    // that recording whose every value is first marked -1, so what the walk
    // reads from a row is told apart from what it stores.
    // [[Rcpp::export]]
    Rcpp::List widening_walk() {
      ode::Solver<Widening> run = kept_solver();
      run.advance_adaptive({0.0, 0.5});
      Widening& sys = run.get_system_ref();
      std::vector<double> before(sys.ode_size()), widened;
      sys.ode_state(before.begin());
      ode::apply_insertion(sys, 0.5, before.begin(), widened);
      run.set_state_from_system();
      run.push_insertion();
      run.advance_adaptive({0.5, 1.0});

      std::vector<Record> marked(run.recording().begin(), run.recording().end());
      for (Record& row : marked) {
        for (Widening::solved_values& v : row.solved) v.found = -1.0;
      }
      ode::Solver<Widening> walk = kept_solver();
      walk.advance_recorded(std::span<const Record>(marked));

      return Rcpp::List::create(Rcpp::Named("run") = describe(run.recording()),
                                Rcpp::Named("walk") = describe(walk.recording()));
    }
  ')
}

test_that("a walk over a recording stores its own rows", {
  compile_widening_interface()
  res <- widening_walk()
  run <- res$run
  walk <- res$walk

  # The rows of the run it walks, the insertion among them, at the same times
  # and widths, reaching the same state.
  expect_equal(sum(run$insertion), 1L)
  expect_identical(walk$insertion, run$insertion)
  expect_identical(bits(walk$time), bits(run$time))
  expect_identical(walk$width, run$width)
  expect_identical(bits(walk$state_end), bits(run$state_end))

  # Each step's slots start as the recorded ones, marked -1, and end holding
  # what the walk's own evaluations stored: the times they ran at.
  step <- rep(!is.nan(run$size), each = 6)
  expect_true(all(walk$seeded[step] == -1))
  expect_identical(bits(walk$found[step]), bits(run$found[step]))
  expect_true(all(is.finite(run$found[step])))
})
