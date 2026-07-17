# The separable field builds a one-sided aggregate A(z) = sum_p a_p(z) B_p(z)
# from a rank-3 kernel over cohorts in descending size order -- exact,
# non-adaptive, no reconstruction. It must reproduce the direct O(N^2) sum for
# the field and its query slope, and its reverse adjoint must be exact: the
# dot-product oracle J v == <v, gradient> (forward tangent through the field vs
# the reverse sweep) to machine precision, needing no finite difference.

test_that("separable field reproduces the kernel, the direct field, and the adjoint oracle", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_separable_field_interface()

  r <- separable_field_demo(eta = 4.0, n = 8)

  # Rank-3 separability: sum_p a_p(z) b_p(x) == kappa(z, x).
  expect_equal(r$sep_recomb, r$sep_direct, tolerance = 1e-10)

  # The assembled field and its query slope match the direct sum over taller
  # sources (the diagonal double zero makes the self term vanish, so A at the
  # tallest cohort is 0).
  expect_equal(r$field_assembled, r$field_direct, tolerance = 1e-9)
  expect_equal(r$slope_assembled, r$slope_direct, tolerance = 1e-9)
  expect_equal(r$field_assembled[1], 0, tolerance = 1e-12)

  # Forward and reverse agree on the value.
  expect_equal(r$value_fwd, r$value_rev, tolerance = 1e-10)

  # The oracle: J v (forward) == <v, gradient> (reverse) to machine precision --
  # the reverse adjoint through the field is exact.
  expect_equal(r$jvp, r$dot_v_grad, tolerance = 1e-11)
})
