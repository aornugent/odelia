# A growing-dimension System (cohorts introduced mid-run, each growing the ODE
# state vector via the solver's resize) must still give a correct reverse-mode
# gradient: the adjoint chain from the metric back through a cohort introduced
# after k resizes must survive the resize. This is the pivotal precondition for
# differentiating the plant SCM (cohorts introduced mid-run). The example checks
# d(sum of final states)/dk against the closed form and finite differences, with
# reserve_state OFF (each introduction reallocates the state vector) and ON.

test_that("reverse gradient survives a mid-run resize (reserve on and off)", {
  testthat::skip_if(is_pkgload_dll(), "native-pointer lifecycle unstable under load_all")
  ensure_growing_resize_interface()

  res <- growing_resize_demo(k = 0.3, delta = 1e-4)

  for (nm in c("no_reserve", "reserve")) {
    r <- res[[nm]]
    # AD matches the closed form to integration accuracy -- the tape survived the
    # resizes (with reserve off this includes vector reallocations).
    expect_equal(r$ad, r$analytic, tolerance = 1e-8,
                 info = sprintf("%s: ad=%.10g analytic=%.10g", nm, r$ad, r$analytic))
    # ... and central finite differences of the forward solve.
    expect_equal(r$ad, r$fd, tolerance = 1e-6,
                 info = sprintf("%s: ad=%.10g fd=%.10g", nm, r$ad, r$fd))
  }
})
