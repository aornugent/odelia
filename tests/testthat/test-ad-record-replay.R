# ODELIA-6: record on a double Solver, replay fixed on an active Solver. The
# RelaxationSystem is the first example System that opts into the Replayable hooks
# (Lorenz/leaf_thermal aren't Replayable), so these are the first tests that drive a
# frozen-knot interpolator on the AD path and exercise record_ode_step / replay_step.
# See docs/ad-record-replay.md.

test_that("record->replay reproduces the adaptive value and matches FD (L1 + L2)", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  gain <- 2.0; y0 <- 1.0; Tmax <- 2.0

  res <- Relaxation_record_replay_gradient(
    Relaxation_new(gain, y0), ctrl$ptr, Tmax, frozen = FALSE)

  # The run genuinely takes several adaptive steps (the schedule that gets frozen).
  expect_gt(res$n_steps, 1)

  # L1: replaying advance_fixed on the recorded schedule reproduces the adaptive
  # final value; L2: the frozen-knot interpolator carries the (double) values.
  adaptive <- Relaxation_adaptive_final(Relaxation_new(gain, y0), ctrl$ptr, Tmax)
  expect_equal(res$value, adaptive, tolerance = 1e-4)

  # The gradient (frozen node positions, active values through the interpolator)
  # matches central finite differences of the fully adaptive run.
  eps <- 1e-6
  hi <- Relaxation_adaptive_final(Relaxation_new(gain + eps, y0), ctrl$ptr, Tmax)
  lo <- Relaxation_adaptive_final(Relaxation_new(gain - eps, y0), ctrl$ptr, Tmax)
  fd <- (hi - lo) / (2 * eps)
  expect_equal(res$gradient[[1]], fd, tolerance = 1e-4)
})

test_that("frozen field: cheap path reproduces, derivative through the field is zero (L3)", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  gain <- 2.0; y0 <- 1.0; Tmax <- 2.0

  res <- Relaxation_record_replay_gradient(
    Relaxation_new(gain, y0), ctrl$ptr, Tmax, frozen = TRUE)

  # The mutant path loads the recorded field values as constants and reproduces the
  # resident trajectory cheaply.
  adaptive <- Relaxation_adaptive_final(Relaxation_new(gain, y0), ctrl$ptr, Tmax)
  expect_equal(res$value, adaptive, tolerance = 1e-3)

  # gain enters only through the field; frozen, its derivative is zero by
  # construction (the invasion-gradient property, in miniature).
  expect_equal(res$gradient[[1]], 0, tolerance = 1e-8)
})

test_that("repeated record->replay calls on one system reproduce", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  sys <- Relaxation_new(2.0, 1.0)

  a <- Relaxation_record_replay_gradient(sys, ctrl$ptr, 2.0, frozen = FALSE)
  b <- Relaxation_record_replay_gradient(sys, ctrl$ptr, 2.0, frozen = FALSE)

  expect_equal(b$value, a$value)
  expect_equal(b$gradient, a$gradient)
})
