# The soil+leaf coupled System: the plant-free witness for the design-live pieces
# the static leaf cannot exercise (v3 §5, §3.2). A soil-water ODE whose per-layer
# potential drives the toy leaf's uptake, which is the soil sink -- the two-way
# feedback loop -- integrated through the odelia Solver, with a ci implicit_value
# node inside the rates and a consumer introduced mid-run (the growing tape). The
# reverse gradient of final biomass w.r.t. the hydraulic traits must match a
# re-integrating finite difference.

testthat::test_that("soil+leaf coupled System compiles and reproduces the double value", {
  ensure_soil_leaf_interface(rebuild = FALSE)
  r <- soil_leaf_demo()
  expect_equal(r$value, r$value_double, tolerance = 1e-12)  # active reproduces double
  expect_gt(r$value, 0)
})

testthat::test_that("soil sub-cycle adjoint matches a re-integrating FD (feedback loop)", {
  ensure_soil_leaf_interface(rebuild = FALSE)
  r <- soil_leaf_demo()
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)
  # kmax and the Weibull shape c both reach the metric only THROUGH the soil
  # feedback (theta -> psi_soil -> uptake -> dtheta/dt). c additionally flows via
  # incomplete_gamma's shape channel. Both match the re-integrating FD.
  expect_lt(reld(r$grad_kmax, r$grad_kmax_fd), 1e-5)
  expect_lt(reld(r$grad_c, r$grad_c_fd), 1e-5)
  expect_gt(abs(r$grad_kmax), 1e-6)   # the feedback channel is live
  expect_gt(abs(r$grad_c), 1e-6)
})

testthat::test_that("the tape stays bounded across the growing run", {
  ensure_soil_leaf_interface(rebuild = FALSE)
  r <- soil_leaf_demo()
  # A whole 3-segment run with resize, a ci node per stage, and the soil feedback
  # records a modest tape (a few MB) -- linear in steps, not the OOM blow-up.
  expect_gt(r$tape_bytes, 0)
  expect_lt(r$tape_bytes, 50e6)
})
