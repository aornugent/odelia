# RIF-3: the active twin (and its tape) are cached on the double Solver and reused
# across gradient calls, while the recording is read fresh from the immutable double
# System on every call. RelaxationSystem is the first Replayable (has_recording)
# System to drive this -- Lorenz records nothing past L1, so its cache can never
# replay a *stale* schedule. That anti-staleness is what RIF-3 is really about.
# See docs/ad-record-replay.md §5 and odelia#12.

test_that("reused twin reproduces, and a fresh recording is picked up per call (RIF-3)", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  d <- Relaxation_Solver_new(Relaxation_new(2.0, 1.0), ctrl$ptr)

  # Record to Tmax = 2, then two gradient calls: the second reuses the cached twin
  # and tape and must reproduce the first exactly -- reuse is transparent.
  n1 <- Relaxation_record(d, Tmax = 2.0)
  expect_gt(n1, 1)
  g1  <- Relaxation_replay_gradient(d, frozen = FALSE)
  g1b <- Relaxation_replay_gradient(d, frozen = FALSE)
  expect_equal(g1b$value, g1$value)
  expect_equal(g1b$gradient, g1$gradient)

  # Anti-staleness: a FRESH double pass (a longer schedule) is picked up on the next
  # call -- the reused twin is handed the new recording, not replayed against the
  # stale Tmax = 2 one. The value matches a fresh adaptive run to 3.0.
  n2 <- Relaxation_record(d, Tmax = 3.0)
  expect_gt(n2, n1)
  g2 <- Relaxation_replay_gradient(d, frozen = FALSE)
  expect_false(isTRUE(all.equal(g2$value, g1$value)))

  adaptive3 <- Relaxation_adaptive_final(Relaxation_new(2.0, 1.0), ctrl$ptr, 3.0)
  expect_equal(g2$value, adaptive3, tolerance = 1e-4)
})

test_that("live and frozen replays share one reused twin, mode chosen per call (RIF-3)", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  d <- Relaxation_Solver_new(Relaxation_new(2.0, 1.0), ctrl$ptr)
  Relaxation_record(d, Tmax = 2.0)

  # Live then frozen on the SAME cached twin: the field mode is a per-call choice.
  live   <- Relaxation_replay_gradient(d, frozen = FALSE)
  frozen <- Relaxation_replay_gradient(d, frozen = TRUE)

  # Both reproduce the trajectory; the frozen field's derivative is zero (the
  # invasion-gradient property), while live carries the real self-feedback gradient.
  expect_equal(frozen$value, live$value, tolerance = 1e-3)
  expect_equal(frozen$gradient[[1]], 0, tolerance = 1e-8)
  expect_gt(abs(live$gradient[[1]]), 1e-6)
})

test_that("replay before record is a clear error, not a stale read (RIF-3)", {
  testthat::skip_if(is_pkgload_dll(), "AD workflow needs the installed DLL, not load_all.")

  ctrl <- odelia:::OdeControl$new()
  d <- Relaxation_Solver_new(Relaxation_new(2.0, 1.0), ctrl$ptr)
  expect_error(Relaxation_replay_gradient(d), "record")
})
