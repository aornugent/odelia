#!/usr/bin/env Rscript
for (d in c("out","fig")) dir.create(d, showWarnings = FALSE)
# Figures for the analysis. Base-R scientific plots; Okabe-Ito colourblind-safe
# categorical palette (fixed assignment per method), single-hue depth ramp for
# soil layers, stacked panels instead of any dual axis.
suppressMessages(library(odelia))
inc <- system.file("include", package="odelia"); so <- file.path(system.file("libs", package="odelia"), "odelia.so")
Sys.setenv(PKG_CPPFLAGS=paste0("-I",inc)); Sys.setenv(PKG_LIBS=shQuote(normalizePath(so)))
Rcpp::sourceCpp("multirate_runner.cpp")
D <- "."
rain <- read.csv(file.path(D,"out/rainfall.csv"))$rain_mm; Tend<-length(rain); daily<-0:Tend

## palette (Okabe-Ito) ------------------------------------------------------
col_rk   <- "#0072B2"; col_ro <- "#D55E00"; col_mr <- "#009E73"; col_mri <- "#CC79A7"
col_rain <- "#4292c6"; ink <- "#222222"; muted <- "#8a8a8a"
depth_ramp <- colorRampPalette(c("#c6dbef","#08306b"))(5)
theme <- function() par(mgp=c(2.1,0.6,0), tcl=-0.3, las=1, cex.axis=0.9, cex.lab=1.0,
                        col.axis=ink, col.lab=ink, fg=muted, family="sans")

## ---- FIG 2: soil moisture + canopy trajectory (pulse-driven dynamics) ------
tr <- global_run(rain, daily, 50L, 1e-8, 1e-10, "rkck", 1.0, 4.0)
png(file.path(D,"fig/soil_trajectory.png"), width=1150, height=760, res=120)
layout(matrix(1:3,3,1), heights=c(1,1.5,1)); par(oma=c(3,0,2,0)); theme()
par(mar=c(0.4,4.2,0,1))
plot(daily[-1], rain, type="h", col=col_rain, lwd=1.3, xaxt="n", xlab="",
     ylab="rain (mm/d)", bty="n"); abline(v=seq(0,Tend,365), col="grey88", lty=3)
mtext("Semi-arid forcing drives pulsed wetting and long dry-downs", 3, line=0.6, cex=0.95, col=ink, font=2)
par(mar=c(0.4,4.2,0,1))
matplot(daily, tr$theta, type="l", lty=1, lwd=1.8, col=depth_ramp, xaxt="n",
        xlab="", ylab=expression(theta~"(m"^3*" m"^-3*")"), bty="n")
abline(h=0.428, col=muted, lty=2); abline(h=0.12, col=muted, lty=3)
abline(v=seq(0,Tend,365), col="grey88", lty=3)
text(Tend*0.995, 0.428, "saturation", pos=1, col=muted, cex=0.75)
text(Tend*0.995, 0.12, "wilting", pos=3, col=muted, cex=0.75)
legend("topright", legend=paste("layer",1:5,c("(top)","","","","(deep)")),
       col=depth_ramp, lwd=2, bty="n", cex=0.8, seg.len=1.4)
par(mar=c(0.4,4.2,0,1))
plot(daily, tr$xbar, type="l", lwd=2, col="#E69F00", xlab="", ylab="canopy status",
     bty="n"); abline(v=seq(0,Tend,365), col="grey88", lty=3)
mtext("day", 1, line=2, outer=TRUE, col=ink)
dev.off(); cat("wrote fig/soil_trajectory.png\n")

## ---- FIG 3: why steps collapse -- accuracy vs stability -------------------
a1 <- readRDS(file.path(D,"out/steplog_stiff1.rds"))
a3 <- readRDS(file.path(D,"out/steplog_stiff3000.rds"))
png(file.path(D,"fig/step_diagnostic.png"), width=1150, height=520, res=120)
par(mfrow=c(1,2)); theme(); par(mar=c(3.6,4.0,2.4,1))
# (a) step size vs time, colored by rainfall day
k <- a1$keep
plot(a1$t[k], a1$h[k], pch=16, cex=0.35, col=adjustcolor(col_rk,0.5), log="y",
     xlab="day", ylab="accepted step size h (day)", bty="n",
     main="(a) RK45 step size over the run")
stormdays <- which(rain>5)-1
abline(v=stormdays, col=adjustcolor(col_rain,0.35), lwd=1)
legend("bottomright", c("accepted step","storm day (>5mm)"), pch=c(16,NA), lty=c(NA,1),
       col=c(col_rk,col_rain), bty="n", cex=0.8)
# (b) histogram of h*|lambda| vs stability ceiling
p1 <- a1$prod[a1$keep]; p3 <- a3$prod[a3$keep]
hist(log10(p1), breaks=40, col=adjustcolor(col_rk,0.5), border=NA, xlim=c(-7,1),
     xlab=expression(log[10](h%.%"|"*lambda*"|")), ylab="accepted steps",
     main="(b) steps sit far below the stability limit", freq=TRUE)
hist(log10(p3), breaks=40, col=adjustcolor(col_ro,0.4), border=NA, add=TRUE)
abline(v=log10(3.3), col=ink, lwd=2, lty=2)
text(log10(3.3), par("usr")[4]*0.9, "explicit stability\nceiling", pos=2, cex=0.78, col=ink)
legend("topleft", c("stiff x1","stiff x3000"), fill=c(adjustcolor(col_rk,0.5),adjustcolor(col_ro,0.4)),
       border=NA, bty="n", cex=0.85)
dev.off(); cat("wrote fig/step_diagnostic.png\n")

## ---- FIG 4: stiffness sweep -- RODAS does not help ------------------------
sA <- read.csv(file.path(D,"out/sweep_stiffness.csv"))
rk <- sA[sA$method=="rkck",]; ro <- sA[sA$method=="rodas",]
png(file.path(D,"fig/stiffness_sweep.png"), width=1150, height=520, res=120)
par(mfrow=c(1,2)); theme(); par(mar=c(3.8,4.2,2.4,1))
plot(rk$stiff, rk$n_steps, type="b", pch=16, lwd=2, col=col_rk, log="x",
     ylim=range(0,rk$n_steps,ro$n_steps), xlab="drainage stiffness  (x K_sat)",
     ylab="accepted steps", bty="n", main="(a) accepted steps vs stiffness")
lines(ro$stiff, ro$n_steps, type="b", pch=17, lwd=2, col=col_ro)
legend("left", c("RK45 (explicit)","RODAS (implicit)"), pch=c(16,17), col=c(col_rk,col_ro),
       lwd=2, bty="n", cex=0.85)
text(30, rk$n_steps[1]*0.75, "flat across 3000x stiffness\n= accuracy-limited", col=ink, cex=0.8, pos=4)
plot(rk$stiff, rk$wall_ms, type="b", pch=16, lwd=2, col=col_rk, log="xy",
     ylim=range(rk$wall_ms,ro$wall_ms), xlab="drainage stiffness  (x K_sat)",
     ylab="wall time (ms)", bty="n", main="(b) wall time vs stiffness")
lines(ro$stiff, ro$wall_ms, type="b", pch=17, lwd=2, col=col_ro)
legend("left", c("RK45","RODAS"), pch=c(16,17), col=c(col_rk,col_ro), lwd=2, bty="n", cex=0.85)
dev.off(); cat("wrote fig/stiffness_sweep.png\n")

## ---- FIG 5: cost vs system size -- dense RODAS blows up; multi-rate flat ---
cs <- read.csv(file.path(D,"out/cost_scaling.csv"))
gk <- cs[cs$method=="rk45",]; gr <- cs[cs$method=="rodas",]
mr <- cs[cs$method=="multirate",]; mi <- cs[cs$method=="multirate_imex",]
png(file.path(D,"fig/cost_scaling.png"), width=1150, height=620, res=120)
theme(); par(mar=c(4.2,4.6,2.8,1))
yr <- range(cs$wall_ms)
plot(gk$N, gk$wall_ms, type="b", pch=16, lwd=2.2, col=col_rk, log="xy",
     xlim=range(cs$N), ylim=yr,
     xlab="system size  N = 5 soil + M canopy", ylab="wall time (ms), full 3-yr run",
     bty="n", main="Cost vs coupled system size: global RODAS is O(N^3); multi-rate is flat in M")
lines(gr$N, gr$wall_ms, type="b", pch=17, lwd=2.2, col=col_ro)
lines(mr$N, mr$wall_ms, type="b", pch=15, lwd=2.2, col=col_mr)
lines(mi$N, mi$wall_ms, type="b", pch=18, lwd=2.2, col=col_mri)
# annotate the RODAS blow-up and the flat multi-rate line
text(gr$N[2], gr$wall_ms[2], sprintf("%.0f s at N=205", gr$wall_ms[3]/1000), pos=2, col=col_ro, cex=0.78)
text(mr$N[nrow(mr)], mr$wall_ms[nrow(mr)], "flat in M", pos=3, col=col_mr, cex=0.8)
legend("left", c("global RK45  (O(N)/step, explicit)","global RODAS (dense O(N^3)/step)",
       "multi-rate (soil sub-cycled, RK45)","multi-rate + RODAS on 5x5 soil (IMEX)"),
       pch=c(16,17,15,18), col=c(col_rk,col_ro,col_mr,col_mri), lwd=2, bty="n", cex=0.82)
dev.off(); cat("wrote fig/cost_scaling.png\n")
cat("all figures done\n")
