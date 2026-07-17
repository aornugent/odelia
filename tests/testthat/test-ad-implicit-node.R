# register_implicit differentiates a scalar inner solve F(y; p) = 0 by the
# implicit function theorem -- dy/dp = -(dF/dp)/(dF/dy) formed by
# forward-differentiating the residual at the root -- without recording the
# iteration. The reverse gradient must match the analytic derivative and a
# re-solve finite difference, plain double returns just the root, and the
# forward-vs-reverse dot-product oracle holds.

test_that("register_implicit gives an exact IFT gradient across double, forward, and reverse", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_implicit_node_interface()

  r <- implicit_node_demo(a = 0.5, b = 1.0, va = 0.4, vb = -0.9)

  # Plain double: the node is just the root, no derivative machinery.
  expect_equal(r$y_double, r$y_star, tolerance = 1e-12)

  # The reverse gradient of g = y^2 matches the analytic IFT derivative.
  expect_equal(r$grad_a, r$grad_a_analytic, tolerance = 1e-9)
  expect_equal(r$grad_b, r$grad_b_analytic, tolerance = 1e-9)

  # The IFT partial matches a re-solve finite difference of the root.
  expect_equal(r$dyda_analytic, r$dyda_fd, tolerance = 1e-7)

  # The oracle: J v (forward) == <v, gradient> (reverse) to machine precision.
  expect_equal(r$jvp, r$dot_v_grad, tolerance = 1e-11)
})
