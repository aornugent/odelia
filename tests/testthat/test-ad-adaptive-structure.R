# An adaptively chosen node set is a function of plain values, so building one at
# an active scalar picks the same nodes as building it plain. That is what lets a
# replayed step rebuild an adaptive background from the state it starts from,
# instead of carrying node positions forward from the pass that discovered them.
#
# The second test is a regression guard on how the refiner predicts midpoints. It
# uses a plain-valued copy of the interpolant; predicting at the active scalar
# instead re-solves the coefficient band once per refinement pass and records
# every solve, which cost ~6x the tape for identical nodes, values and
# derivatives. The bound below fails if that comes back.

testthat::test_that("an adaptive node set is the same built plain or active", {
  ensure_adaptive_structure_interface(rebuild = FALSE)
  r <- adaptive_structure_probe()

  expect_gt(r$n_nodes_plain, 20)                    # the refiner really subdivided
  expect_equal(r$n_nodes_active, r$n_nodes_plain)
  expect_equal(r$node_mismatches, 0L)
  expect_identical(r$node_max_diff, 0)              # bit-identical, not merely close
})

testthat::test_that("the interpolated value and its derivative are unaffected", {
  ensure_adaptive_structure_interface(rebuild = FALSE)
  r <- adaptive_structure_probe()
  reld <- function(a, b) abs(a - b) / (abs(b) + 1e-30)

  expect_identical(r$value_active, r$value_plain)
  # Against the same nodes populated directly, and against a re-refining central
  # difference, which shares no machinery with the tape.
  expect_lt(reld(r$deriv, r$deriv_on_plain_nodes), 1e-14)
  expect_lt(reld(r$deriv, r$deriv_fd), 1e-6)
})

testthat::test_that("refinement is not recorded", {
  ensure_adaptive_structure_interface(rebuild = FALSE)
  r <- adaptive_structure_probe()

  # Building on already-chosen nodes is the floor: one target evaluation per node
  # and one coefficient solve. Refining costs one more solve, not one per pass.
  expect_lt(r$bytes / r$bytes_on_plain_nodes, 2)
})

testthat::test_that("node agreement holds across tolerances and base grids", {
  ensure_adaptive_structure_interface(rebuild = FALSE)
  for (tol in c(1e-4, 1e-6, 1e-8)) {
    for (nb in c(9L, 17L)) {
      r <- adaptive_structure_probe(atol = tol, rtol = tol, nbase = nb)
      expect_equal(r$n_nodes_active, r$n_nodes_plain)
      expect_identical(r$node_max_diff, 0)
      expect_lt(r$bytes / r$bytes_on_plain_nodes, 2)
    }
  }
})
