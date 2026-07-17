# Multirate (MRI-GARK) stepper on the two-rate demonstrator. The system splits
# into a fast block (2 reservoirs) and a slow block (M modes); the macro step
# advances the slow block with an explicit Runge-Kutta while sub-cycling the fast
# block over each leg with the existing adaptive solver.

# empirical convergence ratios of max|MRI - reference| as the macro step halves
convergence_ratios <- function(k, M, table, T_end, Hs, tol) {
  checkt <- seq(0, T_end, by = max(Hs))
  ref <- odelia:::two_rate_reference(k, M, checkt, tol)$states
  errs <- vapply(Hs, function(H) {
    tt <- seq(0, T_end, by = H)
    st <- odelia:::two_rate_mri(k, M, table, tt, tol)$states
    max(abs(st[match(checkt, tt), ] - ref))
  }, numeric(1))
  errs[-length(errs)] / errs[-1]
}

testthat::test_that("macro step reproduces the single-rate reference", {
  k <- 40; M <- 15; tt <- seq(0, 20, by = 1.0)
  ref <- odelia:::two_rate_reference(k, M, tt, 1e-10)
  r <- odelia:::two_rate_mri(k, M, "erk33a", tt, 1e-10)
  expect_equal(dim(r$states), dim(ref$states))
  expect_true(all(is.finite(r$states)))
  # order-3 macro step at a daily grid: close to the fully-resolved solution
  expect_lt(max(abs(r$states - ref$states)), 5e-2)
})

testthat::test_that("with an inert fast block the macro step is a pure ERK", {
  # k = 0 freezes the fast block, so the slow block is exact: x_j = x0_j exp(-w_j t)
  M <- 6; T_end <- 4
  omega <- 0.02 + 0.18 * ((0:(M - 1)) / (M - 1))
  x0 <- 1.0 + 0.5 * cos(pi * (0:(M - 1)) / M)
  checkt <- seq(0, T_end, by = 1.0)
  analytic <- t(sapply(checkt, function(t) c(0, 0, x0 * exp(-omega * t))))
  Hs <- c(1, 0.5, 0.25, 0.125)
  for (tab in c("heun", "kutta3", "erk33a")) {
    errs <- vapply(Hs, function(H) {
      tt <- seq(0, T_end, by = H)
      st <- odelia:::two_rate_mri(0.0, M, tab, tt, 1e-12)$states
      max(abs(st[match(checkt, tt), ] - analytic))
    }, numeric(1))
    ratios <- errs[-length(errs)] / errs[-1]
    order <- odelia:::two_rate_mri(0.0, M, tab, checkt, 1e-12)$order
    expect_gt(min(ratios), 2^order * 0.85)   # halving H cuts error by ~2^order
  }
})

testthat::test_that("coupled order matches the table order when resolved", {
  # k = 1: fast and slow comparable, coupling fully resolved -> textbook order
  ratios_heun <- convergence_ratios(1, 6, "heun", 4, c(1, 0.5, 0.25, 0.125), 1e-12)
  ratios_erk <- convergence_ratios(1, 6, "erk33a", 4, c(1, 0.5, 0.25, 0.125), 1e-12)
  expect_gt(min(ratios_heun), 3.5)   # order 2 -> ratio ~4
  expect_gt(min(ratios_erk), 6.0)    # order 3 -> ratio ~8
})

testthat::test_that("macro step stays consistent when the fast block is stiff", {
  # k = 50: fast block ~50x faster; error still falls with H toward the reference
  ratios <- convergence_ratios(50, 8, "erk33a", 6, c(0.5, 0.25, 0.125), 1e-11)
  expect_true(all(ratios > 1.2))     # monotone convergence under stiffness
  k <- 50; M <- 8; tt <- seq(0, 6, by = 0.125)
  ref <- odelia:::two_rate_reference(k, M, tt, 1e-11)$states
  st <- odelia:::two_rate_mri(k, M, "erk33a", tt, 1e-11)$states
  expect_lt(max(abs(st - ref)), 1e-2)
})

testthat::test_that("fast sub-cycle cost is independent of the slow block size", {
  # the headline: fast-substep count barely moves as the slow block grows 40x
  k <- 50; tt <- seq(0, 30, by = 1.0)
  nf <- vapply(c(10, 100, 400),
               function(M) odelia:::two_rate_mri(k, M, "erk33a", tt, 1e-8)$n_fast,
               numeric(1))
  expect_lt(abs(nf[3] - nf[1]) / nf[1], 0.05)
})
