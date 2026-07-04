# RIF-1: gradients move onto the DOUBLE solver handle. Solver_value_and_gradient
# builds the active replay internally (RIF-2 rebind), so R never holds an active
# solver. Verify it returns exactly what the active-handle Solver_fit produced,
# and that an optimiser loop over the double handle recovers the true parameters.

make_target <- function(true_pars, ctrl) {
  lz <- LorenzSystem$new(true_pars[1], true_pars[2], true_pars[3])
  lz$set_initial_state(c(1, 1, 1), t0 = 0)
  runner <- Lorenz_Solver$new(lz$ptr, ctrl$ptr)
  runner$advance_adaptive(c(0, 1))
  times <- runner$times()
  hist <- runner$history()
  list(times = times,
       target = as.matrix(hist[, c("x", "y", "z")]),
       obs_index = which(times %in% hist$time))
}

test_that("value_and_gradient on the double handle matches the active fit", {
  testthat::skip_if(is_pkgload_dll(), "Skipping AD workflow in pkgload load_all sessions due unstable native-pointer lifecycle.")

  true_pars <- c(sigma = 10.0, R = 28.0, b = 8.0 / 3.0)
  guess <- c(sigma = 12.0, R = 30.0, b = 3.0)
  ctrl <- odelia:::OdeControl$new()
  tgt <- make_target(true_pars, ctrl)

  # Active handle (the pre-RIF-1 path).
  lz_a <- LorenzSystem$new(guess[1], guess[2], guess[3])
  lz_a$set_initial_state(c(1, 1, 1), t0 = 0)
  ad <- Lorenz_Solver$new(lz_a$ptr, ctrl$ptr, active = TRUE)
  Solver_set_target(ad$ptr, tgt$times, tgt$target, tgt$obs_index, active = TRUE)
  f_active <- Solver_fit(ad$ptr, params = guess)

  # Double handle (RIF-1): the active replay is built inside the call.
  lz_d <- LorenzSystem$new(guess[1], guess[2], guess[3])
  lz_d$set_initial_state(c(1, 1, 1), t0 = 0)
  d <- Lorenz_Solver$new(lz_d$ptr, ctrl$ptr)
  Solver_set_target(d$ptr, tgt$times, tgt$target, tgt$obs_index)
  vg <- Solver_value_and_gradient(d$ptr, params = guess)

  expect_equal(vg$value, f_active$loss, tolerance = 1e-12)
  expect_equal(vg$gradient, f_active$gradient, tolerance = 1e-12)
})

test_that("an optimiser over the double handle recovers Lorenz parameters", {
  testthat::skip_if(is_pkgload_dll(), "Skipping AD workflow in pkgload load_all sessions due unstable native-pointer lifecycle.")

  true_pars <- c(sigma = 10.0, R = 28.0, b = 8.0 / 3.0)
  guess <- c(sigma = 12.0, R = 30.0, b = 3.0)
  ctrl <- odelia:::OdeControl$new()
  tgt <- make_target(true_pars, ctrl)

  lz <- LorenzSystem$new(guess[1], guess[2], guess[3])
  lz$set_initial_state(c(1, 1, 1), t0 = 0)
  d <- Lorenz_Solver$new(lz$ptr, ctrl$ptr)  # ONE ordinary (double) solver
  Solver_set_target(d$ptr, tgt$times, tgt$target, tgt$obs_index)

  # value + gradient share one recording; memoise so fn/gr don't each re-run it.
  cache_p <- NULL; cache <- NULL
  vg <- function(p) {
    if (!identical(p, cache_p)) { cache <<- Solver_value_and_gradient(d$ptr, params = p); cache_p <<- p }
    cache
  }
  res <- optim(guess,
               fn = function(p) vg(p)$value,
               gr = function(p) vg(p)$gradient,
               method = "L-BFGS-B", control = list(maxit = 100))

  expect_equal(res$convergence, 0)
  expect_lt(res$value, 1e-10)
  expect_equal(unname(res$par), unname(true_pars), tolerance = 1e-6)
})
