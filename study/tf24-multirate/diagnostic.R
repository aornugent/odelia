#!/usr/bin/env Rscript
# Why do RK45's steps collapse? For every accepted step, compare the actual step
# size h to the explicit stability ceiling h_stab = C/|lambda|, where |lambda| is
# the dominant (drainage) Jacobian eigenvalue at that state. If h << h_stab the
# step is accuracy-limited (an implicit method cannot help); if h ~ h_stab it is
# stability-limited (the regime RODAS is built for).
source("compile.R"); compile_soil_runner()
rain <- read.csv("out/rainfall.csv")$rain_mm
Tend <- length(rain)

# parameters (match soil_runner.cpp) for the Jacobian eigenvalue estimate
theta_sat<-0.428; K_sat<-163.0411; n_psi<-6.57; theta_res<-1e-2; theta_wilt<-0.12
dz<-1500/5; p_drain<-2*n_psi+3; root<-c(0.35,0.28,0.20,0.12,0.05); t_pot<-4.0
# dominant local relaxation rate |d(dtheta_i/dt)/dtheta_i| = (dK/dth + d_uptake/dth)/dz
lambda_max <- function(th_row, stiff) {
  r <- pmax(th_row/theta_sat, 0)
  dK <- p_drain * stiff * K_sat / theta_sat * r^(p_drain-1)         # dK/dtheta
  s  <- pmin(pmax((th_row-theta_res)/(theta_wilt-theta_res),0),1)
  dstress <- ifelse(s>0 & s<1, (6*s-6*s*s)/(theta_wilt-theta_res), 0)
  dUp <- t_pot*root*dstress                                          # d uptake/dtheta
  max(abs(dK + dUp)/dz)
}
C_STAB <- 3.3   # RKCK real-axis linear-stability radius (|h*lambda| ceiling)

analyse <- function(stiff) {
  sl <- soil_steplog(rain, Tend, M=0L, tol_rel=1e-6, tol_abs=1e-8,
                     method="rkck", stiff=stiff, t_pot=4.0)
  t<-sl[,"t"]; h<-sl[,"h"]; th<-sl[,c("th0","th1","th2","th3","th4")]
  lam <- apply(th, 1, lambda_max, stiff=stiff)
  hstab <- C_STAB/pmax(lam,1e-30)
  # a step is "boundary" (forced to a day edge) if it lands within 1e-9 of an integer
  is_edge <- abs(t-round(t))<1e-6
  prod <- h*lam                        # h*|lambda|; ~C_STAB if stability-limited
  # exclude forced day-boundary micro-steps from the classification
  keep <- !is_edge & h>0
  stab_lim <- keep & (prod > 0.5*C_STAB)     # near the stability ceiling
  acc_lim  <- keep & (prod <= 0.5*C_STAB)
  list(t=t,h=h,lam=lam,prod=prod,keep=keep,stab=stab_lim,acc=acc_lim,
       th=th,is_edge=is_edge, hstab=hstab)
}

cat("=== step-limitation diagnostic (RK45, M=0), full horizon ===\n")
rain_day <- rain
for (stiff in c(1, 300, 3000)) {
  a <- analyse(stiff)
  ns <- sum(a$keep)
  cat(sprintf("\nstiff=%g: %d substantive steps (excl. day-edge)\n", stiff, ns))
  cat(sprintf("  median h*|lambda| = %.3g   (stability ceiling ~%.1f)\n",
              median(a$prod[a$keep]), C_STAB))
  cat(sprintf("  steps stability-limited (h*|lambda|>%.1f): %d (%.1f%%)\n",
              0.5*C_STAB, sum(a$stab), 100*sum(a$stab)/ns))
  cat(sprintf("  steps accuracy-limited:                    %d (%.1f%%)\n",
              sum(a$acc), 100*sum(a$acc)/ns))
  # where are the smallest steps? correlate with rainfall of that day
  o <- order(a$h[a$keep])[1:min(50,ns)]
  small_t <- a$t[a$keep][o]
  small_rain <- rain_day[pmin(floor(small_t)+1, length(rain_day))]
  cat(sprintf("  smallest-50 steps: median rain that day = %.1f mm  (overall wet-day rate %.0f%%)\n",
              median(small_rain), 100*mean(rain_day>0)))
  cat(sprintf("  smallest-50 steps: %d of 50 fall on days with rain>5mm (storm-driven accuracy)\n",
              sum(small_rain>5)))
  if (stiff==1) saveRDS(a, "out/steplog_stiff1.rds")
  if (stiff==3000) saveRDS(a, "out/steplog_stiff3000.rds")
}
cat("\nInterpretation: a flat step count across a 3000x stiffness increase, with\n")
cat("h*|lambda| sitting far below the stability ceiling, means the controller is\n")
cat("bounding *truncation error* on rapid real features (storm wetting fronts,\n")
cat("daily forcing kinks), not damping a stiff fast mode. RODAS cannot enlarge\n")
cat("these steps because there is no stability limit to relax.\n")
