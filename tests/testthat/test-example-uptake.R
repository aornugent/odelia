# T6 uptake arbitrage (Slice 3b-iii) on the uptake demonstrator. Over a macro leg
# the fast block reads the expensive coupling a(u) refreshed as an affine model
# a0 + J*(u - anchor); a trust monitor re-captures it only when it drifts. These
# tests are the engine gate that mirrors the offline 3b-ii falsifier: the cheap
# probe-free monitor must (1) track the accuracy of a full-resolve macro run,
# (2) spend far fewer expensive-coupling evaluations than one-per-micro-step and
# stay within a small factor of the oracle count, and (3) admit an exact
# record->replay reverse-mode gradient.

testthat::test_that("affine-refresh macro run matches the full-resolve macro run", {
  # truth macro run = re-capture every micro-step (tol < 0), so any divergence is
  # purely the affine-coupling linearization error -- the clean 3b-ii isolation.
  tt <- seq(0, 0.7, by = 0.1)                         # weekly macro grid (7 legs)
  for (a_scale in c(0.5, 1.0, 2.0)) {
    truth <- odelia:::uptake_mri(a_scale, 5, 6, tt, tol = -1,   nmicro = 40, oracle = TRUE)
    cheap <- odelia:::uptake_mri(a_scale, 5, 6, tt, tol = 1e-2, nmicro = 40, oracle = FALSE)
    expect_true(all(is.finite(cheap$states)))
    err <- max(abs(cheap$states - truth$states))
    expect_lt(err, 5e-3)                              # 3b-ii band (soil accuracy <= 5.5e-4/leg)
  }
})

testthat::test_that("cheap monitor cuts expensive-coupling evals far below baseline", {
  tt <- seq(0, 0.7, by = 0.1)
  cheap <- odelia:::uptake_mri(1.0, 5, 6, tt, tol = 1e-2, nmicro = 40, oracle = FALSE)
  # baseline = one coupling eval per micro-step (the MRI-ancestor cost = 40/leg).
  expect_gt(cheap$baseline / cheap$coupling_evals, 3)   # real reduction (3b-ii: 2.9x-40x)
  # the death mode (re-capture ~every step -> collapse to full-resolve) must NOT occur.
  expect_lt(cheap$coupling_evals, cheap$baseline)
})

testthat::test_that("cheap monitor tracks the oracle re-expansion count", {
  # the oracle triggers on the TRUE a-error (ideal lower bound on captures); the
  # cheap sensitivity-scaled excursion can only match or trail it. 3b-ii spec:
  # within a small factor, given the excursion<->error correlation (0.985).
  tt <- seq(0, 0.7, by = 0.1)
  for (a_scale in c(0.5, 1.0, 2.0)) {
    oracle <- odelia:::uptake_mri(a_scale, 5, 6, tt, tol = 1e-2, nmicro = 40, oracle = TRUE)
    cheap  <- odelia:::uptake_mri(a_scale, 5, 6, tt, tol = 1e-2, nmicro = 40, oracle = FALSE)
    expect_lte(cheap$coupling_evals, 1.7 * oracle$coupling_evals + 1)   # ~1.5x measured
    expect_gte(cheap$coupling_evals, oracle$coupling_evals - 1)         # never cheaper than ideal
  }
})

testthat::test_that("tighter tol spends more captures for more accuracy (the safe knob)", {
  tt <- seq(0, 0.7, by = 0.1)
  loose <- odelia:::uptake_mri(2.0, 5, 6, tt, tol = 5e-2, nmicro = 40, oracle = FALSE)
  tight <- odelia:::uptake_mri(2.0, 5, 6, tt, tol = 1e-3, nmicro = 40, oracle = FALSE)
  expect_gt(tight$coupling_evals, loose$coupling_evals)   # tol trades speed for accuracy
})

testthat::test_that("reverse-mode gradient through the uptake inner matches finite difference", {
  # record->replay: the double pass fixes the re-expansion indices; the active pass
  # replays them under the tape, so the adjoint is the exact discrete gradient of
  # the scheme as run (captures how a0/J at each re-expansion depend on the state).
  tt <- seq(0, 0.7, by = 0.1)
  for (a_scale in c(0.5, 1.0, 2.0)) {
    g <- odelia:::uptake_gradient(a_scale, 5, 6, tt, tol = 1e-2, nmicro = 40, eps_fd = 1e-6)
    expect_length(g$grad_adjoint, 1 + 5 + 6)          # a_scale + initial [u(5); x(6)]
    expect_lt(max(abs(g$grad_adjoint - g$grad_fd)), 1e-6)
    expect_gt(max(abs(g$grad_adjoint)), 1e-3)         # a non-trivial gradient
  }
})

testthat::test_that("adjoint is independent of the finite-difference step", {
  tt <- seq(0, 0.7, by = 0.1)
  g1 <- odelia:::uptake_gradient(1.0, 5, 6, tt, tol = 1e-2, nmicro = 40, eps_fd = 1e-5)
  g2 <- odelia:::uptake_gradient(1.0, 5, 6, tt, tol = 1e-2, nmicro = 40, eps_fd = 1e-7)
  expect_lt(max(abs(g1$grad_adjoint - g2$grad_adjoint)), 1e-11)
})
