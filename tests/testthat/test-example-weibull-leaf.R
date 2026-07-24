# The self-contained Weibull-leaf example: a plant-free miniature of the TF24
# leaf built only from odelia primitives (incomplete_gamma + implicit_value),
# exercising the exact composition that made TF24 hard -- Weibull hydraulics, a
# nested inner ci solve, an outer p* optimum, and the envelope asymmetry between
# the (stationary) profit and the (non-stationary) soil uptake. Every gradient is
# checked against a re-optimising finite difference (the Gate-0 anchor).

# reld between the AD and re-optimising-FD gradient vectors, named by trait.
channel_reld <- function(r, name) {
  ad <- r[[paste0(name, "_ad")]]
  fd <- r[[paste0(name, "_fd")]]
  stats::setNames(abs(ad - fd) / (abs(fd) + 1e-30), r$traits)
}

testthat::test_that("weibull leaf example compiles", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
})

testthat::test_that("weibull leaf sits at an interior optimum", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  r <- weibull_leaf_demo()
  expect_gt(r$p_star, 0.4)                 # collar tension exceeds soil potential
  expect_lt(abs(r$dW_dp_at_pstar), 1e-4)   # stationary => interior, not a bound
})

testthat::test_that("dp*/dtrait matches re-optimising FD (implicit_value node)", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  reld <- channel_reld(weibull_leaf_demo(), "dpstar")
  expect_true(all(reld < 1e-4), info = paste(names(reld), signif(reld, 3), collapse = "  "))
})

testthat::test_that("incomplete_gamma shape channel (c) carries the gradient", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  r <- weibull_leaf_demo()
  # The Weibull shape c flows only through incomplete_gamma's d/da (series)
  # channel; a non-zero, FD-matching gradient proves that channel is live.
  c_idx <- match("c", r$traits)
  expect_gt(abs(r$dpstar_ad[c_idx]), 1e-6)
  expect_lt(channel_reld(r, "dpstar")[["c"]], 1e-4)
  expect_lt(channel_reld(r, "duptake")[["c"]], 1e-3)
})

testthat::test_that("envelope: dW/dtrait matches FD with the dp* term vanishing", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  reld <- channel_reld(weibull_leaf_demo(), "dprofit")
  expect_true(all(reld < 1e-4), info = paste(names(reld), signif(reld, 3), collapse = "  "))
})

testthat::test_that("envelope asymmetry: dE_up/dtrait needs dp* and matches FD", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  # Uptake is not stationary in p*, so its trait gradient carries dp*/dtrait.
  reld <- channel_reld(weibull_leaf_demo(), "duptake")
  expect_true(all(reld < 1e-3), info = paste(names(reld), signif(reld, 3), collapse = "  "))
})

# --- The bound / fold regime (design 4.3, "the one genuinely new concept") ----
testthat::test_that("bound regime: p* pins at p_crit, off the interior optimum", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  b <- weibull_leaf_bound_demo()
  expect_true(b$bound_binds)                 # unconstrained optimum exceeds p_crit
  expect_equal(b$p_star, b$p_crit)           # leaf pinned at the hydraulic bound
  expect_gt(abs(b$dW_dp_at_pstar), 0.1)      # NOT stationary -- genuinely bound
})

testthat::test_that("bound regime: dp*/dtrait flows through the branch-death node", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  r <- weibull_leaf_bound_demo()
  reld <- channel_reld(r, "dpstar")
  expect_true(all(reld < 1e-4), info = paste(names(reld), signif(reld, 3), collapse = "  "))
  # p_crit depends only on the shape c; the other channels are structurally zero.
  expect_gt(abs(r$dpstar_ad[match("c", r$traits)]), 1e-6)
  expect_equal(r$dpstar_ad[match("kmax", r$traits)], 0)
})

testthat::test_that("bound regime: dprofit carries dp* (no envelope cancellation)", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  r <- weibull_leaf_bound_demo()
  # dW/dp != 0 at the bound, so unlike the interior case the profit gradient is
  # NOT the direct partial alone -- it carries dp*/dtrait, on every channel.
  reld <- channel_reld(r, "dprofit")
  expect_true(all(reld < 1e-4), info = paste(names(reld), signif(reld, 3), collapse = "  "))
  expect_true(all(channel_reld(r, "duptake") < 1e-4))
})

# --- Two-layer soil feedback (design 5, the spatial sign-error channel) --------
testthat::test_that("soil: two active layers, p* out-tensions both", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  s <- weibull_leaf_soil_demo()
  expect_true(s$both_active)
  expect_lt(abs(s$dW_dp_at_pstar), 1e-4)     # interior optimum with two soil sinks
})

testthat::test_that("soil: p* and per-layer sinks match FD across all six traits", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  s <- weibull_leaf_soil_demo()
  # dp* and the (non-stationary) uptake channels are limited by the re-optimising
  # FD reference; the envelope profit gradient is exact.
  expect_true(all(channel_reld(s, "dpstar") < 2e-4))
  expect_true(all(channel_reld(s, "dprofit") < 1e-4))
  expect_true(all(channel_reld(s, "dEup") < 2e-4))
  # Per-layer sinks: the spatial psi0/psi1 channel is where a transposed sign
  # historically flipped the gradient. Each layer's sink matches FD.
  expect_true(all(channel_reld(s, "dE0") < 2e-4))
  expect_true(all(channel_reld(s, "dE1") < 2e-4))
})

testthat::test_that("soil: the spatial psi channels are live and correctly signed", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  s <- weibull_leaf_soil_demo()
  # A drier layer (higher psi_i) reduces that layer's uptake: dE_i/dpsi_i < 0.
  i0 <- match("psi0", s$traits); i1 <- match("psi1", s$traits)
  expect_lt(s$dE0_ad[i0], 0)
  expect_lt(s$dE1_ad[i1], 0)
})

# --- The 3-deep nest with the psi_stem inversion node restored (design 4.2) -----
testthat::test_that("stem: psi_stem is a genuine inversion root below the collar", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  s <- weibull_leaf_stem_demo()
  expect_true(s$psi_stem_below_collar)       # distinct from the collar tension p*
  expect_gt(s$psi_stem, 0.4)                  # above soil potential
  expect_lt(abs(s$dW_dp_at_pstar), 1e-4)      # interior optimum of the full nest
})

testthat::test_that("stem: p*, profit, and psi_stem match FD through the full nest", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  s <- weibull_leaf_stem_demo()
  # Every trait threads p -> psi_stem (inversion node) -> ci (root node); each of
  # the three outputs is checked against a re-optimising FD across all four traits.
  expect_true(all(channel_reld(s, "dpstar") < 2e-4))
  expect_true(all(channel_reld(s, "dprofit") < 1e-4))
  expect_true(all(channel_reld(s, "dpsistem") < 2e-4))
  # The inversion channel must be live on the shape trait c (incomplete_gamma d/dc).
  expect_gt(abs(s$dpsistem_ad[match("c", s$traits)]), 1e-3)
})

# --- Tape-memory bound: confirm the design addresses the TF24 OOM (design 4/9) -
testthat::test_that("tape footprint is independent of the inner solver's iterations", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  # The design claim: implicit_value solves OFF the tape, so tape size does not
  # depend on how many iterations the inner solve takes.
  node <- lapply(c(30, 60, 120, 240),
                 function(it) weibull_leaf_tape_profile(1, it, record_solver = FALSE))
  ops <- vapply(node, function(r) r$ops, numeric(1))
  expect_equal(length(unique(ops)), 1L)   # exactly constant across iteration counts
  # The recorded (on-tape solve) path, by contrast, grows with iterations.
  rec <- lapply(c(30, 240),
                function(it) weibull_leaf_tape_profile(1, it, record_solver = TRUE))
  expect_gt(rec[[2]]$ops, 3 * rec[[1]]$ops)
})

testthat::test_that("per-solve tape cost stays small vs the recorded (OOM) path", {
  ensure_weibull_leaf_interface(rebuild = FALSE)
  node1  <- weibull_leaf_tape_profile(1,    60, FALSE)
  node1k <- weibull_leaf_tape_profile(1000, 60, FALSE)
  rec1   <- weibull_leaf_tape_profile(1,    60, TRUE)
  rec1k  <- weibull_leaf_tape_profile(1000, 60, TRUE)
  node_per <- (node1k$ops - node1$ops) / 999
  rec_per  <- (rec1k$ops  - rec1$ops)  / 999
  expect_gt(rec_per, 10 * node_per)       # recorded solve is >10x the per-solve cost
  expect_lt(node_per, 2000)               # node path: a few thousand ops per solve
  # ... and the node path also stays CORRECT: the recorded bisection carries p* as a
  # frozen-endpoint affine combination, so its trait gradient collapses to zero.
  expect_gt(abs(node1$grad_kmax), 1e-3)
  expect_equal(rec1$grad_kmax, 0)
})
