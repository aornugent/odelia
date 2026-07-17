#!/usr/bin/env Rscript
# Forward demonstration on the TF24-structure surrogate: (1) the nested collar solve behaves
# (divergent psi, layers switch off as it dries); (2) CONCEPT — mode-flagged multi-rate matches a
# tight global RK45 reference and converges under macro refinement; (3) STABILITY — completes and
# matches across drought->monsoon with bounded steps, through the divergent read + u_min floor.
here<-"/home/user/plant-dev/scripts/tf24-multirate"; dir.create(file.path(here,"fig"),showWarnings=FALSE)
Rcpp::sourceCpp(file.path(here,"tf24_forward_runner.cpp"))
cat("=== compiled TF24-structure forward runner ===\n")

gen_rain <- function(seed, ndays=365, p01b=0.09, p11b=0.38, shape=0.6, scaleb=11, occ=1) {
  set.seed(seed); doy<-(seq_len(ndays)-1)%%365; season<-(1+cos(2*pi*(doy-15)/365))/2
  p01<-pmin(0.9,(0.010+p01b*season^1.5)*occ); p11<-pmin(0.95,0.15+p11b*season^1.5)
  wet<-logical(ndays); for(t in 2:ndays) wet[t]<-runif(1)<(if(wet[t-1])p11[t] else p01[t])
  scale<-4+scaleb*season; rain<-numeric(ndays); rain[wet]<-rgamma(sum(wet),shape=shape,scale=scale[wet])
  rain[rain<0.1]<-0; round(rain,2)
}
scen <- list(drought=gen_rain(5,occ=0.18,scaleb=6), dry=gen_rain(7,occ=0.45,scaleb=7),
             semiarid=gen_rain(24), wet=gen_rain(11,occ=1.8,scaleb=16),
             monsoon=gen_rain(2,occ=0.7,shape=0.35,scaleb=45))

## (1) collar-solve sanity: divergent psi + layer on/off as soil dries
cat("\n(1) nested collar solve across a drying profile (uniform theta), demand D=3 mm/day:\n")
th<-c(0.40,0.25,0.15,0.10,0.06,0.03,0.015)
pr<-collar_probe(th, 3.0, 5.0)
cat(sprintf("%-8s %10s %10s %8s %9s\n","theta","psi(MPa)","P(MPa)","n_active","E_tot"))
for(j in seq_along(th)) cat(sprintf("%-8.3f %10.3g %10.3f %8.0f %9.3f\n", th[j],pr$psi[j],pr$P[j],pr$n_active[j],pr$Etot[j]))
cat("  (psi diverges as theta->residual; collar P drops; layers switch off; uptake becomes supply-limited)\n")

## (2) CONCEPT: multi-rate vs tight global reference (semi-arid), macro refinement
cat("\n(2) concept: mode-flagged multi-rate vs tight global RK45 (semi-arid, M=50)\n")
M<-50L; Tend<-365; daily<-0:Tend
ref<-global_run(scen$semiarid, daily, M, 1e-9,1e-11, 5.0)
errf<-function(r,mac){ idx<-match(mac,daily); max(max(abs(r$theta-ref$theta[idx,,drop=FALSE])), max(abs(r$cbar-ref$cbar[idx]))) }
cat(sprintf("%-8s %12s %12s %10s\n","macro H","max|err|","soil_steps","wall_ms"))
for(H in c(4,2,1)){ mac<-seq(0,Tend,by=H); r<-mrate_run(scen$semiarid, mac, M, 1e-8,1e-10, 5.0)
  cat(sprintf("%-8d %12.2e %12.0f %10.1f\n", H, errf(r,mac), r$soil_steps, r$wall_ms)) }
cat(sprintf("  (global RK45 reference: %.0f steps, %.1f ms)\n", ref$steps, ref$wall_ms))

## (3) STABILITY across drought->monsoon (daily macro)
cat("\n(3) stability across scenarios (M=50, daily macro):\n")
cat(sprintf("%-9s %7s %6s | %-9s %10s %9s %8s %9s\n","scenario","mm/yr","maxmm","complete?","max|err|","soil_stp","min_th","min_P(MPa)"))
traj<-list()
for(sn in names(scen)){ rn<-scen[[sn]]
  rf<-global_run(rn, daily, M, 1e-9,1e-11, 5.0)
  r <-mrate_run(rn, daily, M, 1e-8,1e-10, 5.0)
  idx<-match(daily,daily); err<-max(max(abs(r$theta-rf$theta)),max(abs(r$cbar-rf$cbar)))
  ok<-all(is.finite(r$theta))&&all(is.finite(r$cbar))
  # min collar potential reached (probe at the driest recorded profile)
  minth<-min(r$theta); pr<-collar_probe(rep(minth,1),3.0,5.0)
  cat(sprintf("%-9s %7.0f %6.1f | %-9s %10.1e %9.0f %8.4f %9.1f\n",
      sn, sum(rn), max(rn), ifelse(ok,"YES","NO"), err, r$soil_steps, minth, pr$P[1]))
  traj[[sn]]<-r$theta[,1]
}

## figure: top-layer soil moisture across scenarios (with wilting/saturation guides)
ink<-"#222"; muted<-"#8a8a8a"; cols<-c(drought="#D55E00",dry="#E69F00",semiarid="#0072B2",wet="#009E73",monsoon="#CC79A7")
png(file.path(here,"fig/tf24_forward_stability.png"), width=1100, height=460, res=120)
par(mgp=c(2.1,0.6,0),tcl=-0.3,las=1,mar=c(4,4.4,2.6,1),fg=muted,col.axis=ink,col.lab=ink)
plot(NA,xlim=c(0,Tend),ylim=c(0,0.43),xlab="day",ylab=expression(theta[1]~"(top layer)"),bty="n",
     main="Mode-flagged multi-rate on TF24-structure surrogate (nested collar solve, divergent psi, u_min floor)")
abline(h=c(0.12,0.428),col=muted,lty=3)
for(sn in names(traj)) lines(daily,traj[[sn]],lwd=1.8,col=cols[sn])
legend("topright",legend=names(scen),col=cols[names(scen)],lwd=2,bty="n",cex=0.8)
dev.off(); cat("\nwrote fig/tf24_forward_stability.png\n")
