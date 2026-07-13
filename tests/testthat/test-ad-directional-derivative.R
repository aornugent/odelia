# directional_derivative computes df/dx along one direction by forward-over-reverse
# and keeps the result active on the outer reverse tape, so an outer sweep gives the
# mixed second derivative d(df/dx)/dtheta exactly -- no finite-difference step, and a
# clamp/kink handled correctly (the derivative of a constant is zero). This is the
# tool plant needs for the density-transport term dg/dh (odelia#38, plant#39).

test_that("directional_derivative gives exact dg/dh and its parameter derivative", {
  testthat::skip_if(is_pkgload_dll(), "Skipping AD workflow in pkgload load_all sessions due unstable native-pointer lifecycle.")
  ensure_directional_derivative_interface()

  # Unclamped: dg/dh and d(dg/dh)/db0 both match the analytic values exactly, and
  # here also agree with the finite-difference-of-a-finite-difference (no kink).
  r <- directional_derivative_demo(h0 = 5.0, C = 0.0)
  expect_false(as.logical(r$clamped))
  expect_equal(r$dgdh, r$dgdh_analytic, tolerance = 1e-12)
  expect_equal(r$d2_ad, r$d2_analytic, tolerance = 1e-12)   # exactly 1
  expect_equal(r$d2_ad, r$d2_fd_of_fd, tolerance = 1e-6)

  # In the clamped region (heavy competition drives growth below zero, so it is
  # pinned to zero) the correct derivatives are exactly zero, and forward-over-
  # reverse delivers that without the /eps blow-up the on-tape stencil would incur.
  r2 <- directional_derivative_demo(h0 = 5.0, C = 200.0)
  expect_true(as.logical(r2$clamped))
  expect_equal(r2$dgdh, 0.0, tolerance = 1e-12)
  expect_equal(r2$d2_ad, 0.0, tolerance = 1e-12)
})
