# SPIKE, not a shipped feature: does a step-local reverse sweep over a stored
# trajectory reproduce the whole-run gradient on a GROWING-dimension System?
#
# The backward loop lives in the example file, NOT in odelia, so nothing here commits an
# interface -- the point is to settle the algorithm by execution before any contract is
# designed around it (docs/v3-step-local-adjoint.md §3c/§3d).
#
# The load-bearing test is the last one. A newborn's initial condition either ignores the
# standing state (`constant`, which is what every existing odelia toy has) or reads it
# (`coupled`, which is what plant has -- a newborn's density reads the active stand). With
# a coupled IC the structural change MUST be re-recorded inside a unit; placed between
# units its adjoint is silently lost. Under a constant IC both placements agree, so the
# existing toys cannot detect that error at all.

testthat::test_that("step-local sweep reproduces the whole-run gradient, FD and closed form", {
  ensure_step_local_interface(rebuild = FALSE)
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)
  for (ic in c("constant", "coupled")) {
    for (uk in c("step", "segment")) {
      r <- step_local_demo(k = 0.3, nstep = 10, unit_kind = uk, ic_kind = ic)
      # Against the whole-run tape: the same chain-rule product, reassociated, so this
      # is a round-off agreement rather than an approximation.
      expect_lt(reld(r$grad_step_local, r$grad_whole_run), 1e-12)
      # Against a re-integrating central FD, which shares no machinery with either.
      expect_lt(reld(r$grad_step_local, r$grad_fd), 1e-6)
      if (ic == "constant") {
        expect_lt(reld(r$grad_step_local, r$grad_analytic), 1e-8)
      }
    }
  }
})

testthat::test_that("peak tape is flat in run length, while the whole-run tape is linear", {
  ensure_step_local_interface(rebuild = FALSE)
  peaks <- whole <- numeric(0)
  for (n in c(10, 20, 40, 80)) {
    r <- step_local_demo(k = 0.3, nstep = n, unit_kind = "step", ic_kind = "coupled")
    peaks <- c(peaks, r$peak_bytes_step_local)
    whole <- c(whole, r$peak_bytes_whole_run)
    expect_lt(abs(r$grad_step_local / r$grad_whole_run - 1), 1e-12)
  }
  # The whole point: peak tape does not grow with the run.
  expect_equal(peaks, rep(peaks[1], length(peaks)))
  # ... while the whole-run tape does, so the ratio keeps widening.
  expect_gt(whole[4] / whole[1], 6)
  expect_gt(whole[4] / peaks[4], 100)
})

testthat::test_that("a finer unit gives a smaller peak, and the segment unit grows with the run", {
  ensure_step_local_interface(rebuild = FALSE)
  by_step <- step_local_demo(k = 0.3, nstep = 40, unit_kind = "step", ic_kind = "coupled")
  by_seg  <- step_local_demo(k = 0.3, nstep = 40, unit_kind = "segment", ic_kind = "coupled")
  expect_lt(abs(by_seg$grad_step_local / by_seg$grad_whole_run - 1), 1e-12)
  # A segment holds every step between two introductions, so its tape scales with the
  # schedule's density -- which is why the unit is one ODE step and not one segment.
  expect_gt(by_seg$peak_bytes_step_local / by_step$peak_bytes_step_local, 10)
})

testthat::test_that("a structural change placed BETWEEN units silently loses its adjoint", {
  ensure_step_local_interface(rebuild = FALSE)
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)

  # Constant IC: the newborn's IC does not read the stand, so its adjoint contributes
  # nothing and BOTH placements are right. This is the trap -- every existing odelia toy
  # (growing_resize's 1.0, soil_leaf's W0) has a constant IC, so none of them can fail.
  for (pl in c("inside", "outside")) {
    r <- step_local_demo(k = 0.3, nstep = 10, unit_kind = "step",
                         ic_kind = "constant", intro_placement = pl)
    expect_lt(reld(r$grad_step_local, r$grad_whole_run), 1e-12)
  }

  # Coupled IC (plant's shape): inside is right, outside is wrong -- and wrong quietly,
  # with no error and a plausible magnitude.
  ok <- step_local_demo(k = 0.3, nstep = 10, unit_kind = "step",
                        ic_kind = "coupled", intro_placement = "inside")
  bad <- step_local_demo(k = 0.3, nstep = 10, unit_kind = "step",
                         ic_kind = "coupled", intro_placement = "outside")
  expect_lt(reld(ok$grad_step_local, ok$grad_whole_run), 1e-12)
  expect_gt(reld(bad$grad_step_local, bad$grad_whole_run), 1e-3)
  expect_true(is.finite(bad$grad_step_local))            # no error, just wrong
  expect_equal(sign(bad$grad_step_local), sign(ok$grad_step_local))  # plausible, too
})
