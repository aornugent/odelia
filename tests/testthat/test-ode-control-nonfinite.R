# The adaptive controller's step decision on a non-finite error estimate. The
# error ratios are reduced with std::max, which does not propagate NaN: a NaN
# followed by a finite ratio is dropped, so the largest measured error comes from
# the components that happened to stay finite. Both readings accept the step --
# unchanged when the NaN survives to the end (rmax is NaN, so neither the
# too-large nor the too-small test fires), and grown when it is dropped and the
# survivors are small. An accepted non-finite state enters the trajectory, and a
# replay pinned to that grid differentiates it.

test_that("a non-finite error estimate rejects the step, wherever it sits", {
  ctrl <- odelia:::OdeControl_new()
  odelia:::OdeControl_set_controls(ctrl,
                          tol_abs = 1e-8, tol_rel = 1e-8,
                          a_y = 1.0, a_dydt = 0.0,
                          step_size_min = 1e-12, step_size_max = 10.0,
                          step_size_initial = 1e-6)

  h <- 0.1
  decide <- function(yerr) {
    n <- length(yerr)
    odelia:::OdeControl_adjust_step_size(ctrl, ord = 5, step_size = h,
                                y = rep(1.0, n), yerr = yerr,
                                dydt = rep(0.0, n))
  }

  # Every position of the NaN: last (it survives the reduction), first and
  # middle (it is dropped, and the surviving ratios are small enough to grow).
  for (yerr in list(c(1e-12, NaN),
                    c(NaN, 1e-12),
                    c(1e-12, NaN, 1e-12))) {
    r <- decide(yerr)
    expect_true(r$shrank)
    expect_lt(r$step_size_next, h)
  }

  # Inf is the same case.
  expect_true(decide(c(1e-12, Inf))$shrank)

  # A zero error ratio is not this case and must still accept: errlevel is
  # floored by tol_abs, so its denominator is nonzero whenever a tolerance is,
  # and 0/D0 is an exactly-met step rather than a non-finite one.
  expect_false(decide(c(0.0, 0.0))$shrank)
})

test_that("the finite decisions are unchanged", {
  ctrl <- odelia:::OdeControl_new()
  odelia:::OdeControl_set_controls(ctrl,
                          tol_abs = 1e-8, tol_rel = 1e-8,
                          a_y = 1.0, a_dydt = 0.0,
                          step_size_min = 1e-12, step_size_max = 10.0,
                          step_size_initial = 1e-6)
  h <- 0.1

  # Well inside tolerance: accept and propose a larger step.
  small <- odelia:::OdeControl_adjust_step_size(ctrl, 5, h, rep(1.0, 2), rep(1e-12, 2),
                                       rep(0.0, 2))
  expect_false(small$shrank)
  expect_gt(small$step_size_next, h)

  # Far outside: reject, and no more than a factor of five down.
  big <- odelia:::OdeControl_adjust_step_size(ctrl, 5, h, rep(1.0, 2), rep(1.0, 2),
                                     rep(0.0, 2))
  expect_true(big$shrank)
  expect_equal(big$step_size_next, h * 0.2)
})
