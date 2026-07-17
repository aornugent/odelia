# The scalar value guards: smooth_positive is a C-infinity max(0, x) with a
# declared corner radius (so a clamped rate stays differentiable at the kink),
# and is_finite reads an active value without a strip. smooth_positive's
# derivative must be exact, and the dot-product oracle J v == <v, gradient>
# confirms the reverse adjoint through it needs no finite difference.

test_that("smooth_positive is an exact differentiable clamp and is_finite reads active NaN", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_value_guards_interface()

  r <- value_guards_demo(r = 1e-2)

  # As the radius -> 0, smooth_positive recovers the sharp max(0, x).
  expect_equal(r$sp_tiny, r$relu, tolerance = 1e-6)

  # The reverse gradient matches the analytic derivative 0.5(1 + x/sqrt(x^2+r^2)).
  expect_equal(r$grad, r$grad_ref, tolerance = 1e-12)

  # The oracle: J v (forward) == <v, gradient> (reverse) to machine precision.
  expect_equal(r$jvp, r$dot_v_grad, tolerance = 1e-11)

  # is_finite on the active type: NaN is false, a finite value is true.
  expect_false(r$nan_is_finite)
  expect_true(r$finite_is_finite)
})
