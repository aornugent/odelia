# decide records a value-dependent branch on the double pass and replays it on
# the differentiated pass, so the control flow the tape is built on cannot flip
# when the state is active or a parameter is perturbed. The gradient is then the
# one-sided, frozen-branch derivative -- which a frozen-schedule finite
# difference (the same recorded branches, gain +/- h) reproduces. diagnostic
# reads the final state off the tape as a plain double.

test_that("decide replays the recorded branch and gives the frozen-schedule gradient", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_decide_interface()

  r <- decide_demo(gain = 2.0, Tmax = 3.0, nsteps = 30)

  # The run actually switches branch -- there is a schedule worth recording.
  expect_gt(r$n_branches, 1)

  # Replaying the recorded branches on the active pass reproduces the double value.
  expect_equal(r$value, r$value_double, tolerance = 1e-10)

  # The gradient is the frozen-branch derivative: a frozen-schedule finite
  # difference (same recorded branches) matches it.
  expect_equal(r$grad, r$fd_frozen, tolerance = 1e-6)

  # diagnostic is a dead read: the plain value of the (active) final state.
  expect_equal(r$monitored, r$value, tolerance = 1e-12)
})
