#!/usr/bin/env Rscript
# Main benchmark: RK45 (rkck) vs RODAS on the TF24 soil model under the semi-arid
# rainfall sequence. Two sweeps:
#   (A) drainage-stiffness sweep, soil-only (M=0): does RODAS help, and when?
#   (B) coupled-size sweep (stiff=1): how does dense-RODAS cost scale with N=5+M?
# Effort = accepted steps + wall time + RHS evals; accuracy = max error vs a tight
# reference (RODAS, tol 1e-10). Daily kink-split grid (fair within-day comparison).

source("compile.R"); compile_soil_runner()
rain <- read.csv("out/rainfall.csv")$rain_mm
Tend <- length(rain); grid <- 0:Tend
TOL_R <- 1e-6; TOL_A <- 1e-8

ref_final <- function(M, stiff) {
  r <- soil_bench(rain, grid, M=as.integer(M), tol_rel=1e-10, tol_abs=1e-12,
                  method="rodas", stiff=stiff, t_pot=4.0)
  list(theta=r$theta, x0=r$x0)
}
run1 <- function(M, stiff, method, ref) {
  r <- soil_bench(rain, grid, M=as.integer(M), tol_rel=TOL_R, tol_abs=TOL_A,
                  method=method, stiff=stiff, t_pot=4.0)
  err <- max(abs(r$theta - ref$theta))
  data.frame(M=M, stiff=stiff, method=method, n_steps=r$n_steps,
             wall_ms=r$wall_ms, rhs_dbl=r$rhs_double, rhs_twin=r$rhs_twin,
             rhs_tot=r$rhs_double+r$rhs_twin, err_theta=err)
}

cat("=== (A) drainage-stiffness sweep, soil-only (M=0), full 3-year horizon ===\n")
resA <- list()
for (stiff in c(1, 30, 300, 3000)) {
  ref <- ref_final(0, stiff)
  a <- run1(0, stiff, "rkck",  ref)
  b <- run1(0, stiff, "rodas", ref)
  resA[[length(resA)+1]] <- a; resA[[length(resA)+1]] <- b
  cat(sprintf("stiff=%5g | RKCK  steps=%6.0f wall=%7.1fms rhs=%8.0f err=%.1e\n",
              stiff, a$n_steps, a$wall_ms, a$rhs_tot, a$err_theta))
  cat(sprintf("          | RODAS steps=%6.0f wall=%7.1fms rhs=%8.0f err=%.1e   (RODAS/RKCK: steps %.2fx, wall %.2fx)\n",
              b$n_steps, b$wall_ms, b$rhs_tot, b$err_theta, b$n_steps/a$n_steps, b$wall_ms/a$wall_ms))
}
dfA <- do.call(rbind, resA)
write.csv(dfA, "out/sweep_stiffness.csv", row.names=FALSE)

cat("\n=== (B) coupled-size sweep (stiff=1), N = 5 + M ===\n")
resB <- list()
for (M in c(0, 20, 100, 400)) {
  ref <- ref_final(M, 1)
  a <- run1(M, 1, "rkck",  ref)
  b <- run1(M, 1, "rodas", ref)
  resB[[length(resB)+1]] <- a; resB[[length(resB)+1]] <- b
  cat(sprintf("N=%4d (M=%3d) | RKCK  wall=%8.1fms steps=%6.0f rhs=%9.0f\n",
              5+M, M, a$wall_ms, a$n_steps, a$rhs_tot))
  cat(sprintf("              | RODAS wall=%8.1fms steps=%6.0f rhs=%9.0f (twin=%.0f)  wall %.2fx RKCK\n",
              b$wall_ms, b$n_steps, b$rhs_tot, b$rhs_twin, b$wall_ms/a$wall_ms))
}
dfB <- do.call(rbind, resB)
write.csv(dfB, "out/sweep_coupled.csv", row.names=FALSE)
cat("\nDONE. wrote out/sweep_stiffness.csv, out/sweep_coupled.csv\n")
