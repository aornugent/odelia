# compute_jvp is the forward-mode (tangent) Jacobian-vector product of an ODE solve:
# one augmented run, no tape, transparent across the growing-dimension resize. Paired
# with the reverse gradient it gives the FD-free adjoint dot-product oracle
# <J v, u> = <v, Jᵀ u>: for a scalar output (u = 1) the forward directional derivative
# J v must equal <v, gradient> to machine precision -- a correctness check needing no
# perturbation and no inner-solve re-run. Exercised on a growing two-parameter System
# with a closed form (odelia#38; the oracle plant#39 will use to certify census gradients).

test_that("compute_jvp matches the reverse gradient (dot-product oracle) and the closed form", {
  testthat::skip_if(is_pkgload_dll(), "Skipping AD workflow in pkgload load_all sessions due unstable native-pointer lifecycle.")
  ensure_jvp_oracle_interface()

  for (dir in list(c(1, 0), c(0, 1), c(0.37, 0.71), c(-1.3, 2.0))) {
    r <- jvp_oracle_demo(k = 0.3, b = 1.0, vk = dir[1], vb = dir[2])

    # Forward and reverse agree on the value, and both reproduce the closed form.
    expect_equal(r$value_fwd, r$value_rev, tolerance = 1e-12)
    expect_equal(r$value_rev, r$value_analytic, tolerance = 1e-8)

    # The reverse gradient is exact.
    expect_equal(r$grad_k, r$grad_k_analytic, tolerance = 1e-8)
    expect_equal(r$grad_b, r$grad_b_analytic, tolerance = 1e-8)

    # The oracle: forward J v == <v, gradient> to machine precision (this is the
    # FD-free self-consistency of the two AD modes), and both == the analytic
    # directional derivative.
    expect_equal(r$jvp, r$dot_v_grad, tolerance = 1e-11)
    expect_equal(r$jvp, r$jvp_analytic, tolerance = 1e-8)
  }
})
