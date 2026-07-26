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

# A quantity defined by an equation solved off tape, read inside the rates, is the
# shape a leaf operating point has in the real model. The node is rebuilt every time
# a unit is re-recorded, so it is the case most likely to break: an implicit lift
# grafts a correction whose value is discarded, and doing that once per unit rather
# than once per run must give the same adjoint.
testthat::test_that("an implicitly defined rate re-records exactly, per unit", {
  ensure_step_local_interface(rebuild = FALSE)
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)
  for (ic in c("constant", "coupled")) {
    r <- step_local_demo(k = 0.3, nstep = 10, unit_kind = "step", ic_kind = ic,
                         intro_placement = "inside", rate_kind = "implicit")
    expect_lt(reld(r$grad_step_local, r$grad_whole_run), 1e-12)
    expect_lt(reld(r$grad_step_local, r$grad_fd), 1e-6)
  }
})

testthat::test_that("peak tape stays flat with an implicit node in the rates", {
  ensure_step_local_interface(rebuild = FALSE)
  peaks <- whole <- numeric(0)
  for (n in c(10, 20, 40, 80)) {
    r <- step_local_demo(k = 0.3, nstep = n, unit_kind = "step", ic_kind = "coupled",
                         intro_placement = "inside", rate_kind = "implicit")
    peaks <- c(peaks, r$peak_bytes_step_local)
    whole <- c(whole, r$peak_bytes_whole_run)
    expect_lt(abs(r$grad_step_local / r$grad_whole_run - 1), 1e-12)
  }
  # Flat in run length, while the whole-run tape is linear in it -- so the ratio
  # keeps widening rather than settling at some constant factor.
  expect_equal(peaks, rep(peaks[1], length(peaks)))
  expect_gt(whole[4] / whole[1], 6)
  expect_gt(whole[4] / peaks[4], 100)
})

# The real reduction is a census vector, so a unit has to serve several adjoint rows.
# Recording once and sweeping once per row is what matters for cost: if it works, a
# three-output census Jacobian costs the same tape as a scalar and only more sweeps.
testthat::test_that("several Jacobian rows come off one re-recording per unit", {
  ensure_step_local_interface(rebuild = FALSE)
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)
  for (rk in c("closed_form", "implicit")) {
    r <- step_local_jacobian_demo(k = 0.3, nstep = 10, ic_kind = "coupled",
                                  rate_kind = rk)
    expect_length(r$d_k_step_local, 2)
    # Both rows against the whole-run tape, and against a re-integrating difference.
    for (i in 1:2) {
      expect_lt(reld(r$d_k_step_local[i], r$d_k_whole_run[i]), 1e-12)
      expect_lt(reld(r$d_k_step_local[i], r$d_k_fd[i]), 1e-6)
    }
    # The rows are genuinely different reductions, so this is not one row twice.
    expect_gt(abs(r$d_k_step_local[1] - r$d_k_step_local[2]), 1e-3)
  }
})

testthat::test_that("a second output row costs no extra tape", {
  ensure_step_local_interface(rebuild = FALSE)
  one <- step_local_demo(k = 0.3, nstep = 10, unit_kind = "step", ic_kind = "coupled",
                         intro_placement = "inside", rate_kind = "implicit")
  two <- step_local_jacobian_demo(k = 0.3, nstep = 10, ic_kind = "coupled",
                                  rate_kind = "implicit")
  # Two rows sweep the same recording twice rather than recording twice, so peak
  # tape is set by the unit, not by the number of outputs.
  expect_lt(two$peak_bytes_step_local, 1.2 * one$peak_bytes_step_local)
})

# Reusing one tape rewound between units was the obvious optimisation over building a
# fresh one each time. It does not work, and this records why so it is not retried:
# rewinding to a marked position keeps the gradient exact but does NOT release the
# tape, so peak memory grows with the run -- which is the one thing the whole design
# exists to prevent. The per-unit tape construction is a real cost and it is the price
# of the bound, not an oversight.
testthat::test_that("rewinding one tape does not bound peak memory; a fresh tape does", {
  ensure_step_local_interface(rebuild = FALSE)
  fresh <- reused <- numeric(0)
  for (n in c(20, 40, 80, 160)) {
    r <- step_local_demo(k = 0.3, nstep = n, unit_kind = "step", ic_kind = "coupled",
                         intro_placement = "inside", rate_kind = "implicit")
    fresh <- c(fresh, r$peak_bytes_step_local)
    reused <- c(reused, r$peak_bytes_reused_tape)
    # Both are exact -- this is a memory result, not a correctness one.
    expect_lt(abs(r$grad_reused_tape / r$grad_whole_run - 1), 1e-12)
  }
  expect_equal(fresh, rep(fresh[1], length(fresh)))   # flat
  expect_gt(reused[4] / reused[1], 4)                  # grows with the run
  expect_gt(reused[4] / fresh[4], 20)
})
