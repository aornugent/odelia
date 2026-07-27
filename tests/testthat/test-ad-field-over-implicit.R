# A separable_field assembled over implicit_value source weights -- TF24's shape.
#
# K93 and FF16 build the field from closed-form source weights, so nothing measured so
# far exercises a field whose cumulative sums are taken over IFT-derived quantities.
# This is that witness. See inst/examples/field_over_implicit_interface.cpp.

test_that("a field over implicit source weights differentiates correctly", {
  ensure_field_over_implicit_interface()

  r <- field_over_implicit_demo()
  expect_equal(r$severed, FALSE)

  # The active and the pure-double path are the same body, so the value must agree
  # exactly: implicit_value returns y* bit-identically, introducing no shift.
  expect_identical(r$value, r$value_double)

  # Every channel against a central FD. amp/eta reach J through the field's assembly;
  # k/theta/n_psi reach it ONLY through the leaf solve, so these three are the ones
  # that test the composition rather than the field.
  names(r$grad) <- r$names
  names(r$fd) <- r$names
  for (nm in r$names) {
    expect_equal(r$grad[[nm]], r$fd[[nm]],
                 tolerance = 1e-6,
                 info = sprintf("channel %s: reverse %.12g vs FD %.12g",
                                nm, r$grad[[nm]], r$fd[[nm]]))
  }

  # No channel may be silently zero -- that is what severance looks like, and it is
  # the failure this whole line of work exists to catch.
  for (nm in r$names) {
    expect_gt(abs(r$grad[[nm]]), 0)
  }
})

test_that("the amp channel matches an analytic identity, independent of the FD", {
  ensure_field_over_implicit_interface()

  # amp factors out of both cumulative sums and u* does not depend on it, so
  # dA/damp = A/amp exactly. An FD cannot distinguish a correct IFT partial from a
  # merely smooth one; this pins the field assembly on its own.
  r <- field_over_implicit_demo()
  a <- field_over_implicit_amp_identity()
  names(r$grad) <- r$names
  expect_equal(r$grad[["amp"]], a$dJ_damp_analytic, tolerance = 1e-12)
})

test_that("severing the leaf solve collapses exactly the coupled channels", {
  ensure_field_over_implicit_interface()

  live <- field_over_implicit_demo(sever_uptake = FALSE)
  sev <- field_over_implicit_demo(sever_uptake = TRUE)
  names(live$grad) <- live$names
  names(sev$grad) <- sev$names

  # The control must change the derivative and nothing else. to_passive keeps the
  # value, so a differing value would mean the control is not clean.
  expect_identical(live$value, sev$value)
  expect_equal(sev$severed, TRUE)

  # k, theta and n_psi reach J only through the solve, so freezing it must zero them
  # exactly. This is the assertion that makes the test a witness rather than a
  # smoke test: it proves the live gradient's k/theta/n_psi content comes from the
  # IFT partials and from nowhere else.
  for (nm in c("k", "theta", "n_psi")) {
    expect_identical(sev$grad[[nm]], 0)
    expect_gt(abs(live$grad[[nm]]), 1e-8)
  }

  # amp and eta do not pass through the solve, so severance must leave them
  # bit-identical. If these moved, the control would be changing the field too and
  # the collapse above would prove nothing about the coupling.
  for (nm in c("amp", "eta")) {
    expect_identical(sev$grad[[nm]], live$grad[[nm]])
  }
})

test_that("the composition holds as the population grows", {
  ensure_field_over_implicit_interface()

  # The field's cost and its cumulative depth both scale with the source count, and
  # each source adds an implicit_value node. Exactness must not degrade with depth.
  for (n in c(2L, 5L, 12L, 40L)) {
    r <- field_over_implicit_demo(n_src = n)
    names(r$grad) <- r$names
    names(r$fd) <- r$names
    for (nm in r$names) {
      expect_equal(r$grad[[nm]], r$fd[[nm]],
                   tolerance = 1e-5,
                   info = sprintf("n_src = %d, channel %s", n, nm))
    }
  }
})

test_that("a drier soil still differentiates through every source", {
  ensure_field_over_implicit_interface()

  # theta is the shared coupling, and psi = theta^-n is stiff at small theta. A
  # gradient that is exact only in the comfortable middle is not a witness for a
  # water-limited model.
  for (th in c(0.5, 0.25, 0.1, 0.05)) {
    r <- field_over_implicit_demo(theta = th)
    names(r$grad) <- r$names
    names(r$fd) <- r$names
    expect_equal(r$grad[["theta"]], r$fd[["theta"]],
                 tolerance = 1e-5,
                 info = sprintf("theta = %g", th))
    expect_gt(abs(r$grad[["theta"]]), 0)
  }
})
