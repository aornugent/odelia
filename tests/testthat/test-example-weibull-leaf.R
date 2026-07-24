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
