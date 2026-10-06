# Tests that the step a System is handed has a fourth-order dense output ending
# on the step's solution; that a block is integrated in pieces between its sign
# changes, once where its sign value changes sign across a step and twice where it
# dips inside one, a stage holding the other sign or not; that a System that
# splits nothing runs bit for bit as one that cannot split; that a walk takes the
# recorded run's splits; and that the sweep refuses a run that split.

compile_split_interface <- function() {
  ensure_ode_interface_loaded()

  odelia_so <- .odelia_test_cache$odelia_so
  withr::local_envvar(
    PKG_CPPFLAGS = odelia_cppflags(odelia_include_dir()),
    PKG_LIBS = shQuote(normalizePath(odelia_so, winslash = "/", mustWork = TRUE))
  )
  Rcpp::sourceCpp(code = '
    // [[Rcpp::plugins(cpp20)]]
    #include <Rcpp.h>
    #include <algorithm>
    #include <cmath>
    #include <vector>
    #include <odelia/ode_solver.hpp>

    using namespace odelia;
    using record_type = std::vector<ode::split_block<ode::no_solved_values>>;

    // A toy System splits as plant does: each block whose sign value changes sign
    // in the step is integrated in pieces that meet at its sign changes, its rates
    // read from what it samples of the rest of the state at the five sample
    // fractions. Block p is component p. `read(state, out)` writes what a block
    // reads; `rate(p, sample, out)` writes the rate of block p and returns its sign
    // value.
    template <class Step, class Read, class Rate>
    bool split_toy(const Step& step, std::size_t blocks, record_type& record,
                   Read&& read, Rate&& rate) {
      std::array<std::vector<double>, 5> samples;
      std::vector<double> state(step.y0.size()), sample, out(1);
      bool sampled = false;
      auto block_rates = [&](std::size_t p, double u, std::vector<double>& r) {
        if (!sampled) {
          for (std::size_t m = 0; m < samples.size(); ++m) {
            step.dense_state(step.sample_fractions[m], 0, state);
            read(state, samples[m]);
          }
          sampled = true;
        }
        step.sample_at(u, samples, sample);
        return rate(p, sample, r);
      };
      for (std::size_t p = 0; p < blocks; ++p) {
        auto changes = step.template sign_changes<ode::no_solved_values>(
          p, [&](double u, ode::no_solved_values*) {
            return block_rates(p, u, out);
          });
        if (changes.empty()) {
          continue;
        }
        ode::split_block<ode::no_solved_values>& block = record.emplace_back();
        block.block = p;
        block.first = p;
        block.state_before_split.assign(1, step.y_end[p]);
        std::vector<double> split_at;
        for (const auto& change : changes) {
          split_at.push_back(change.u);
        }
        block.sign_changes = std::move(changes);
        std::vector<double> own(1);
        step.integrate_pieces(p, split_at,
                              [&](double u, const std::vector<double>&,
                                  std::vector<double>& r) { block_rates(p, u, r); },
                              own);
        step.y_end[p] = own[0];
      }
      return sampled;
    }

    // The toys walk laid out as they ran, so each recorded block is carried in
    // place.
    static void carry_in_place(const record_type& recorded,
                               const std::vector<double>& run_end,
                               std::vector<double>& y) {
      for (const auto& block : recorded) {
        for (std::size_t q = 0; q < block.state_before_split.size(); ++q) {
          const std::size_t i = block.first + q;
          y[i] = (y[i] - block.state_before_split[q]) + run_end[i];
        }
      }
    }

    // y1 = cos t, y2 = -sin t. Its split reads the dense output at each of `u`
    // and the end the step reached, and splits nothing.
    struct Oscillator {
      using value_type = double;
      size_t ode_size() const { return 2; }
      double ode_time() const { return time; }
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
        *it++ = y[1];
        *it++ = -y[0];
        return it;
      }
      void sign_values(std::vector<double>& out) const { out.clear(); }
      template <class Step>
      bool split_sign_changes(const Step& step, record_type&) {
        std::vector<double> out(2);
        dense.clear();
        for (double v : u) {
          step.dense_state(v, 0, out);
          dense.push_back(out);
        }
        step.dense_state(1.0, 0, out);
        dense.push_back(out);
        end = step.y_end;
        return false;
      }
      void take_recorded_splits(const record_type&, const std::vector<double>&,
                                std::vector<double>&) const {}
      std::vector<double> y{1.0, 0.0};
      double time = 0.0;
      std::vector<double> u, end;
      std::vector<std::vector<double>> dense;
    };

    // One step of size h from y = (1, 0), then the dense output at each u, less
    // the solution there; the last row is the end less the dense output at 1.
    // [[Rcpp::export]]
    Rcpp::NumericMatrix oscillator_dense(double h, std::vector<double> u) {
      Oscillator sys;
      sys.u = u;
      ode::Solver<Oscillator> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      solver.advance_fixed({0.0, h});
      const Oscillator& s = solver.get_system_ref();
      Rcpp::NumericMatrix ret(static_cast<int>(u.size()) + 1, 2);
      for (size_t i = 0; i < u.size(); ++i) {
        ret(i, 0) = s.dense[i][0] - std::cos(u[i] * h);
        ret(i, 1) = s.dense[i][1] + std::sin(u[i] * h);
      }
      ret(u.size(), 0) = s.end[0] - s.dense.back()[0];
      ret(u.size(), 1) = s.end[1] - s.dense.back()[1];
      return ret;
    }

    // Four blocks of one component, x_p rising at max(g_p, 0), then z = t and
    // w = t^2 + 2t, which no block holds. g_0 = z - 0.3; g_1 is below zero on
    // (0.55, 0.65), where the stage at 0.6 falls, and g_2 on (0.06, 0.14), before
    // the first stage. g_3 is below zero for w in (0.3, 0.42), t in (0.140, 0.192),
    // so at w = 0.4 in the first stage, but not at w(0.2) = 0.44 on the dense
    // output. Only Kinked<true> can split.
    template <bool Splits>
    struct Kinked {
      using value_type = double;
      static double gate(std::size_t p, double z, double w) {
        return p == 0   ? z - 0.3
               : p == 1 ? (z - 0.6) * (z - 0.6) - 0.0025
               : p == 2 ? (z - 0.1) * (z - 0.1) - 0.0016
                        : (w - 0.3) * (w - 0.42);
      }
      size_t ode_size() const { return 6; }
      double ode_time() const { return time; }
      void reset() {
        y.assign(6, 0.0);
        time = 0.0;
      }
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
        for (std::size_t p = 0; p < 4; ++p) {
          g[p] = gate(p, y[4], y[5]);
          *it++ = std::max(g[p], 0.0);
        }
        *it++ = 1.0;
        *it++ = 2.0 * (y[4] + 1.0);
        return it;
      }
      void sign_values(std::vector<double>& out) const requires Splits {
        out.assign(g.begin(), g.end());
      }
      // A block reads z and w.
      template <class Step>
      bool split_sign_changes(const Step& step, record_type& record)
        requires Splits {
        if (!splits) {
          return false;
        }
        return split_toy(
          step, 4, record,
          [](const std::vector<double>& state, std::vector<double>& out) {
            out.assign(state.begin() + 4, state.begin() + 6);
          },
          [](std::size_t p, const std::vector<double>& sample,
             std::vector<double>& out) {
            const double v = gate(p, sample[0], sample[1]);
            out[0] = std::max(v, 0.0);
            return v;
          });
      }
      void take_recorded_splits(const record_type& recorded,
                                const std::vector<double>& run_end,
                                std::vector<double>& y) const requires Splits {
        carry_in_place(recorded, run_end, y);
      }

      bool splits = true;
      std::vector<double> y{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      std::vector<double> g{0.0, 0.0, 0.0, 0.0};
      double time = 0.0;
    };

    template <bool Splits>
    static ode::Solver<Kinked<Splits>> kinked_solver(bool splits, bool keep) {
      Kinked<Splits> sys;
      sys.splits = splits;
      ode::Solver<Kinked<Splits>> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      solver.set_keep_states(keep);
      solver.reset();
      return solver;
    }

    static std::vector<int> splits_of(const std::vector<std::size_t>& by_block) {
      return std::vector<int>(by_block.begin(), by_block.end());
    }

    // One step pinned over [0, 1]: the state it ends on, the splits by block, and
    // where each block it split changed sign.
    // [[Rcpp::export]]
    Rcpp::List kinked_one_step(bool splits) {
      auto solver = kinked_solver<true>(splits, true);
      solver.advance_fixed({0.0, 1.0});
      Rcpp::List at;
      for (const auto& block : solver.recording().back().solved.split_blocks) {
        std::vector<double> u;
        for (const auto& change : block.sign_changes) {
          u.push_back(change.u);
        }
        at.push_back(u);
      }
      return Rcpp::List::create(Rcpp::Named("state") = solver.state(),
                                Rcpp::Named("splits") = splits_of(solver.splits_by_block()),
                                Rcpp::Named("sign_changes") = at);
    }

    // An adaptive run over [0, 2]: splitting, able to split but not, or unable to;
    // then a walk over the recording of the first.
    // [[Rcpp::export]]
    Rcpp::List kinked_runs() {
      auto split = kinked_solver<true>(true, true);
      split.advance_adaptive({0.0, 2.0});
      auto unsplit = kinked_solver<true>(false, false);
      unsplit.advance_adaptive({0.0, 2.0});
      auto none = kinked_solver<false>(false, false);
      none.advance_adaptive({0.0, 2.0});

      std::vector<ode::step_record<Kinked<true>>> rows(
        split.recording().begin(), split.recording().end());
      auto walk = kinked_solver<true>(true, true);
      walk.advance_recorded(rows);
      std::vector<double> run_states, walk_states;
      for (const auto& row : split.recording()) {
        run_states.insert(run_states.end(), row.state.begin(), row.state.end());
      }
      for (const auto& row : walk.recording()) {
        walk_states.insert(walk_states.end(), row.state.begin(), row.state.end());
      }

      auto run = [](auto& s) {
        return Rcpp::List::create(
          Rcpp::Named("times") = s.times(),
          Rcpp::Named("step_sizes") = s.step_sizes(),
          Rcpp::Named("state") = s.state(),
          Rcpp::Named("splits") = splits_of(s.splits_by_block()));
      };
      return Rcpp::List::create(Rcpp::Named("split") = run(split),
                                Rcpp::Named("unsplit") = run(unsplit),
                                Rcpp::Named("none") = run(none),
                                Rcpp::Named("walk") = run(walk),
                                Rcpp::Named("run_states") = run_states,
                                Rcpp::Named("walk_states") = walk_states);
    }

    // Two blocks of one component, x_p rising at the smooth positive part of g_p,
    // then z rising at one, so z = t. g_0 = z - a changes sign once; g_1 =
    // (z - c)^2 - b dips below zero about c. a, b and c are the parameters, so a
    // sweep moves the sign changes.
    template <typename T>
    struct Turning {
      using value_type = T;
      Turning(T a_ = T(0.3), T b_ = T(0.0025), T c_ = T(0.6))
        : a(a_), b(b_), c(c_) {}
      template <class S2>
      Turning<S2> rebind_from() const {
        Turning<S2> out(S2(util::to_passive(a)), S2(util::to_passive(b)),
                        S2(util::to_passive(c)));
        for (std::size_t i = 0; i < 3; ++i) {
          out.y[i] = S2(util::to_passive(y[i]));
        }
        out.time = time;
        return out;
      }
      std::vector<T*> ad_parameters() { return {&a, &b, &c}; }
      template <class F> void for_each_active(F&& f) {
        f(a); f(b); f(c);
        for (T& v : y) f(v);
        for (T& v : g) f(v);
        for (T& v : dydt) f(v);
      }
      T gate(std::size_t p, const T& z) const {
        if (p == 0) {
          return T(z - a);
        }
        return T((z - c) * (z - c) - b);
      }
      static T turn(const T& v) {
        using std::sqrt;
        return T(0.5 * (v + sqrt(v * v + 0.0025)));
      }
      size_t ode_size() const { return 3; }
      double ode_time() const { return time; }
      void reset() {
        y.assign(3, T(0.0));
        time = 0.0;
      }
      template <typename It> It set_ode_state(It it, double t) {
        for (T& v : y) v = *it++;
        time = t;
        for (std::size_t p = 0; p < 2; ++p) {
          g[p] = gate(p, y[2]);
          dydt[p] = turn(g[p]);
        }
        dydt[2] = 1.0;
        return it;
      }
      template <typename It> It ode_state(It it) const {
        for (const T& v : y) *it++ = v;
        return it;
      }
      template <typename It> It ode_rates(It it) const {
        for (const T& v : dydt) *it++ = v;
        return it;
      }
      void sign_values(std::vector<double>& out) const {
        out.resize(2);
        for (std::size_t p = 0; p < 2; ++p) {
          out[p] = util::to_passive(g[p]);
        }
      }
      // A block reads z alone.
      template <class Step>
      bool split_sign_changes(const Step& step, record_type& record) {
        return split_toy(
          step, 2, record,
          [](const std::vector<double>& state, std::vector<double>& out) {
            out.assign(1, state[2]);
          },
          [this](std::size_t p, const std::vector<double>& sample,
                 std::vector<double>& out) {
            const double v = util::to_passive(gate(p, T(sample[0])));
            out[0] = util::to_passive(turn(T(v)));
            return v;
          });
      }
      void take_recorded_splits(const record_type& recorded,
                                const std::vector<double>& run_end,
                                std::vector<double>& y) const {
        carry_in_place(recorded, run_end, y);
      }

      T a, b, c;
      std::vector<T> y{T(0.0), T(0.0), T(0.0)};
      std::vector<T> g{T(0.0), T(0.0)};
      std::vector<T> dydt{T(0.0), T(0.0), T(0.0)};
      double time = 0.0;
    };

    // Sweep a run over [0, 1] at `pars` that split, which the sweep refuses.
    // [[Rcpp::export]]
    int turning_sweep(std::vector<double> pars) {
      Turning<double> sys(pars[0], pars[1], pars[2]);
      ode::Solver<Turning<double>> solver(sys, ode::OdeControl());
      solver.set_collect(false);
      solver.set_keep_states(true);
      solver.set_state({0.0, 0.0, 0.0}, 0.0);
      solver.advance_fixed({0.0, 1.0});
      ode::adjoint_rows lambda = ode::adjoint_rows::one_row({1.0, 1.0, 1.0});
      ode::adjoint_rows rows(1, 3);
      solver.solve_adjoint(lambda, rows);
      return static_cast<int>(solver.splits_by_block().size());
    }
  ', verbose = FALSE)
}

bits <- function(x) vapply(x, function(v) paste(writeBin(v, raw()), collapse = ""),
                           character(1))

test_that("the dense output is fourth order and ends on the step's solution", {
  compile_split_interface()
  u <- c(0.25, 0.5, 0.75)
  coarse <- oscillator_dense(0.2, u)
  fine <- oscillator_dense(0.1, u)
  # A fourth-order output errs by h^5 inside the step, so at least 32 times less
  # at h / 2; the cosine's leading term vanishes, and it falls 64 times.
  ratio <- abs(coarse[1:3, ]) / abs(fine[1:3, ])
  expect_true(all(ratio > 24))
  expect_true(all(ratio[, 2] < 40))
  expect_true(all(abs(coarse[4, ]) < 1e-15))
  expect_true(all(abs(fine[4, ]) < 1e-15))
})

test_that("one step is integrated in pieces at a sign change and at a dip", {
  compile_split_interface()
  # x_0 = (1 - 0.3)^2 / 2; x_1 is g_1's integral less its part inside the dip,
  # 0.28 / 3 - 0.0025 + 4 * 0.05^3 / 3, and x_2 likewise, 0.730256 / 3 - 0.0016.
  # With s = 1 + t, g_3 = (s^2 - 1.3)(s^2 - 1.42), whose integral is F. Each piece
  # is a polynomial the stepper integrates exactly.
  F <- function(s) s^5 / 5 - 2.72 * s^3 / 3 + 1.846 * s
  x3 <- F(2) - F(1) - (F(sqrt(1.42)) - F(sqrt(1.3)))
  exact <- c(0.245, 0.091, 0.730256 / 3 - 0.0016, x3, 1, 3)
  split <- kinked_one_step(TRUE)
  plain <- kinked_one_step(FALSE)
  expect_equal(split$state, exact, tolerance = 1e-13)
  expect_equal(split$splits, c(1L, 1L, 1L, 1L))
  expect_gt(abs(plain$state[1] - exact[1]), 1e-2)
  expect_gt(abs(plain$state[2] - exact[2]), 1e-4)
  expect_gt(abs(plain$state[3] - exact[3]), 1e-5)
  expect_gt(abs(plain$state[4] - exact[4]), 1e-6)
  expect_length(plain$splits, 0L)
})

test_that("a dip no stage holds is found beside the reading nearest zero", {
  compile_split_interface()
  # g_2 reads 0.0084 at the step's start and at the first stage, within 2% of its
  # readings' spread of 0.8, and is below zero only between them. g_3's first
  # stage reads -0.002, which the dense output does not hold, so its gaps are
  # searched. g_1's dip holds the stage at 0.6, so it is split without a search.
  at <- kinked_one_step(TRUE)$sign_changes
  tol <- 1e-9
  expect_equal(at[[1]], 0.3, tolerance = tol)
  expect_equal(at[[2]], c(0.55, 0.65), tolerance = tol)
  expect_equal(at[[3]], c(0.06, 0.14), tolerance = tol)
  expect_equal(at[[4]], sqrt(c(1.3, 1.42)) - 1, tolerance = tol)
})

test_that("a System that splits nothing runs bit for bit as one that cannot", {
  compile_split_interface()
  res <- kinked_runs()
  expect_identical(bits(res$unsplit$times), bits(res$none$times))
  expect_identical(bits(res$unsplit$step_sizes), bits(res$none$step_sizes))
  expect_identical(bits(res$unsplit$state), bits(res$none$state))
  # Splitting moves the run: x_0 reaches (2 - 0.3)^2 / 2 exactly.
  expect_equal(res$split$state[1], 1.445, tolerance = 1e-13)
  expect_false(identical(res$split$state, res$none$state))
  expect_gt(sum(res$split$splits), 0L)
  expect_length(res$unsplit$splits, 0L)
})

test_that("a walk takes the recorded run's splits and repeats it bit for bit", {
  compile_split_interface()
  res <- kinked_runs()
  expect_identical(bits(res$walk_states), bits(res$run_states))
  # The walk splits nothing of its own.
  expect_length(res$walk$splits, 0L)
})

test_that("the sweep refuses a run that split", {
  compile_split_interface()
  expect_error(turning_sweep(c(0.3, 0.0025, 0.6)),
               "split a block at a sign change")
})
