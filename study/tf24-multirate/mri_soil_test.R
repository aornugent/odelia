#!/usr/bin/env Rscript
# MRI vs Lie-split (Rung 0) on the coupled soil+canopy model: H-refinement coupling
# order against a tight global reference. Confirms MRI (Midpoint order 2, Kutta3
# order 3) beats the first-order Lie split at matched macro step H.
suppressMessages(library(odelia))
inc  <- system.file("include", package="odelia")
so   <- file.path(system.file("libs", package="odelia"), "odelia.so")
here <- "/home/user/plant-dev/scripts/tf24-multirate"
Sys.setenv(PKG_CPPFLAGS=paste0("-I",inc," -I",here))
Sys.setenv(PKG_LIBS=shQuote(normalizePath(so)))
Rcpp::sourceCpp(file.path(here,"multirate_runner.cpp"))
cat("=== compiled (with MRI) ===\n")

rain <- read.csv(file.path(here,"out/rainfall.csv"))$rain_mm
Tend <- 360; M <- 100L; alo<-0.02; ahi<-0.10; TR<-1e-10; TA<-1e-12
daily <- 0:Tend
ref <- global_run(rain, daily, M, 1e-11, 1e-13, "rkck", 1.0, 4.0)
# max error over the whole trajectory (robust to single-point cancellation)
errf <- function(r, macro) {
  idx <- match(macro, daily)
  et <- max(abs(r$theta - ref$theta[idx, , drop=FALSE]))
  ex <- max(abs(r$xbar - ref$xbar[idx]))
  max(et, ex)
}

Hs <- c(8,4,2,1)
cat(sprintf("\ncoupling error (max over trajectory) vs tight reference (M=%d, %d-day horizon):\n", M, Tend))
cat(sprintf("%-14s %10s %10s %10s %10s   %s\n","method","H=8","H=4","H=2","H=1","fitted order"))
run_row <- function(label, fn) {
  e <- sapply(Hs, function(H){ macro<-seq(0,Tend,by=H); r<-fn(macro); errf(r, macro) })
  ord <- coef(lm(log(e) ~ log(Hs)))[2]      # slope of log-err vs log-H = order
  cat(sprintf("%-14s %10.2e %10.2e %10.2e %10.2e   %.2f\n", label, e[1],e[2],e[3],e[4], ord))
  invisible(e)
}
E <- list()
E[["Lie split"]]    <- run_row("Lie split",   function(macro) multirate_run(rain,macro,M,TR,TA,"rkck",1.0,4.0,alo,ahi))
E[["MRI FwdEuler"]] <- run_row("MRI FwdEuler",function(macro) mri_run(rain,macro,M,TR,TA,"fwd_euler",1.0,4.0,alo,ahi))
E[["MRI Midpoint"]] <- run_row("MRI Midpoint",function(macro) mri_run(rain,macro,M,TR,TA,"midpoint",1.0,4.0,alo,ahi))
E[["MRI Kutta3"]]   <- run_row("MRI Kutta3",  function(macro) mri_run(rain,macro,M,TR,TA,"kutta3",1.0,4.0,alo,ahi))

# convergence figure (Okabe-Ito; guide slopes 1/2/3)
png(file.path(here,"fig/mri_convergence.png"), width=1050, height=560, res=120)
par(mgp=c(2.1,0.6,0), tcl=-0.3, las=1, mar=c(4.2,4.4,2.6,1), fg="#8a8a8a", col.axis="#222", col.lab="#222")
cols <- c("Lie split"="#0072B2","MRI FwdEuler"="#999999","MRI Midpoint"="#009E73","MRI Kutta3"="#CC79A7")
pch  <- c("Lie split"=16,"MRI FwdEuler"=4,"MRI Midpoint"=15,"MRI Kutta3"=18)
plot(NA, xlim=rev(range(Hs)), ylim=range(unlist(E)), log="xy", bty="n",
     xlab="macro step H (days)", ylab="coupling error (max over trajectory)",
     main="MRI vs Lie split on the coupled soil+canopy model (semi-arid forcing)")
for (nm in names(E)) lines(Hs, E[[nm]], type="b", lwd=2, col=cols[nm], pch=pch[nm])
# guide slopes anchored at H=1 lower end
gx <- c(1,8)
for (s in 1:3) { gy <- 2e-6*(gx)^s; lines(gx, gy, lty=3, col="#bbbbbb") }
text(5, 2e-6*5^1, "O(H)", col="#999", cex=0.75, pos=3)
text(5, 2e-6*5^2, "O(H²)", col="#999", cex=0.75, pos=3)
text(5, 2e-6*5^3, "O(H³)", col="#999", cex=0.75, pos=3)
abline(v=1, col="#f0c0c0", lty=2); text(1, max(unlist(E)), "kink-aligned\n(H = forcing grid)", col="#c0392b", cex=0.72, pos=2)
legend("bottomright", legend=names(E), col=cols[names(E)], pch=pch[names(E)], lwd=2, bty="n", cex=0.85)
dev.off(); cat("wrote fig/mri_convergence.png\n")

# accuracy per unit work at the kink-aligned daily macro step (H=1)
cat("\naccuracy vs cost at kink-aligned H=1 (daily macro = forcing grid):\n")
macro<-seq(0,Tend,by=1)
lr<-multirate_run(rain,macro,M,TR,TA,"rkck",1.0,4.0,alo,ahi)
cat(sprintf("  Lie split    soil_steps=%.0f wall=%.1fms err=%.2e\n", lr$soil_steps, lr$wall_ms, errf(lr, macro)))
for (tab in c("fwd_euler","midpoint","kutta3")) {
  r<-mri_run(rain,macro,M,TR,TA,tab,1.0,4.0,alo,ahi)
  cat(sprintf("  MRI %-10s soil_steps=%.0f wall=%.1fms err=%.2e  (%.0fx more accurate than Lie)\n",
      tab, r$soil_steps, r$wall_ms, errf(r, macro), errf(lr,macro)/errf(r,macro)))
}
