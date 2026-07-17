# incomplete_gamma(a, x) is the lower incomplete gamma, the exact antiderivative
# behind the Weibull integral. Written in elementary operations, so AD reads its
# value, d/dx (the integrand) and d/da (the shape channel) off the same code. It
# must match pgamma, its d/dx must be the integrand, and the Weibull composition
# G(m) = (b/c) gamma(1/c, (m/b)^c) must have dG/dm equal to the integrand
# exp(-(m/b)^c) exactly (the Leibniz endpoint).

test_that("incomplete_gamma matches pgamma and gives exact value and shape derivatives", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_incomplete_gamma_interface()

  r <- incomplete_gamma_demo(a = 0.4, x = 3.0, b = 2.0, c = 5.0, m = 3.0)

  # Value against the standard library: gamma(a,x) = pgamma(x,a) * gamma(a).
  expect_equal(r$value, pgamma(r$x, r$a) * gamma(r$a), tolerance = 1e-10)

  # d/dx is the integrand x^{a-1} e^{-x} exactly.
  expect_equal(r$d_dx, r$d_dx_ref, tolerance = 1e-10)

  # d/da (the shape channel) matches a finite difference.
  expect_equal(r$d_da, r$d_da_fd, tolerance = 1e-7)

  # Weibull endpoint: dG/dm equals the integrand exp(-(m/b)^c) by Leibniz.
  expect_equal(r$dG_dm, r$dG_dm_ref, tolerance = 1e-10)

  # The differentiated-trait channel dG/dc (through d/da) matches a finite difference.
  expect_equal(r$dG_dc, r$dG_dc_fd, tolerance = 1e-6)

  # Large x: the series still reaches gamma(a, x) within the term cap.
  expect_equal(r$value_large, pgamma(r$x_large, r$a) * gamma(r$a), tolerance = 1e-9)
})
