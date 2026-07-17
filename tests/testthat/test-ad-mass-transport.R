# The density-transport rate is d(log density)/dt = -C - loss, C the neighbour
# secant of the growth rate over the same spacing every reduction weights with.
# Because the two share one operator, C equals the log-rate of the spacing's own
# evolution -- transported as log mass the compression cancels exactly, and its
# reverse adjoint is exact (the dot-product oracle needs no finite difference).

test_that("mass transport cancels the compression and gives an exact adjoint", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_mass_transport_interface()

  r <- mass_transport_demo(n = 8)

  # Cancellation: -rate equals the log-rate of the spacing's evolution at every
  # node -- so log mass transports as -loss with no compression residual.
  expect_equal(r$cancel_rate, r$cancel_ref, tolerance = 1e-12)

  # The neighbour secant reproduces the analytic dg/dx on the interior.
  expect_equal(r$secant, r$gprime_ref, tolerance = 1e-9)

  # Forward and reverse agree on the value.
  expect_equal(r$value_fwd, r$value_rev, tolerance = 1e-10)

  # The oracle: J v (forward) == <v, gradient> (reverse) to machine precision.
  expect_equal(r$jvp, r$dot_v_grad, tolerance = 1e-11)
})
