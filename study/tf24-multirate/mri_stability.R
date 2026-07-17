#!/usr/bin/env Rscript
# Forward stability of the MRI multirate scheme across realistic + challenging rainfall
# scenarios: does it complete without blow-up, and match a tight global reference, with
# bounded step counts? Scenarios span bone-dry drought to monsoon bursts.
suppressMessages(library(odelia))
inc<-system.file("include",package="odelia"); so<-file.path(system.file("libs",package="odelia"),"odelia.so")
here<-"/home/user/plant-dev/scripts/tf24-multirate"; dir.create(file.path(here,"fig"),showWarnings=FALSE)
Sys.setenv(PKG_CPPFLAGS=paste0("-I",inc," -I",here)); Sys.setenv(PKG_LIBS=shQuote(normalizePath(so)))
Rcpp::sourceCpp(file.path(here,"multirate_runner.cpp"))
cat("=== compiled ===\n")

gen_rain <- function(seed, ndays=365, p01b=0.09, p11b=0.38, shape=0.6, scaleb=11, occ=1) {
  set.seed(seed); doy<-(seq_len(ndays)-1)%%365; season<-(1+cos(2*pi*(doy-15)/365))/2
  p01<-pmin(0.9,(0.010+p01b*season^1.5)*occ); p11<-pmin(0.95,0.15+p11b*season^1.5)
  wet<-logical(ndays); for(t in 2:ndays) wet[t]<-runif(1)<(if(wet[t-1])p11[t] else p01[t])
  scale<-4+scaleb*season; rain<-numeric(ndays); rain[wet]<-rgamma(sum(wet),shape=shape,scale=scale[wet])
  rain[rain<0.1]<-0; round(rain,2)
}
scen <- list(
  drought = gen_rain(5,  occ=0.18, scaleb=6),
  dry     = gen_rain(7,  occ=0.45, scaleb=7),
  semiarid= gen_rain(24),
  wet     = gen_rain(11, occ=1.8,  scaleb=16),
  monsoon = gen_rain(2,  occ=0.7,  shape=0.35, scaleb=45)  # rare, very intense bursts
)
M<-50L; Tend<-365; daily<-0:Tend
cat(sprintf("\n%-9s %8s %6s %7s | %-10s %9s %9s %8s\n",
    "scenario","mm/yr","%wet","maxmm","completed?","max|err|","soil_stp","min_theta"))
res<-list(); traj<-list()
for(sn in names(scen)){
  rn<-scen[[sn]]
  ref<-global_run(rn, daily, M, 1e-9,1e-11,"rkck",1.0,4.0)
  r  <-mri_run(rn, daily, M, 1e-8,1e-10,"midpoint",1.0,4.0,0.02,0.10)
  err<-max(abs(r$theta-ref$theta))
  ok <-all(is.finite(r$theta)) && all(is.finite(r$xbar))
  mth<-min(r$theta)
  cat(sprintf("%-9s %8.0f %6.1f %7.1f | %-10s %9.1e %9.0f %8.4f\n",
      sn, sum(rn), 100*mean(rn>0), max(rn), ifelse(ok,"YES","NO"), err, r$soil_steps, mth))
  res[[sn]]<-list(err=err,steps=r$soil_steps,ok=ok,mm=sum(rn)); traj[[sn]]<-r$theta[,1]
}

# figure: top-layer soil moisture across scenarios (all complete, no blow-up)
ink<-"#222"; muted<-"#8a8a8a"
cols<-c(drought="#D55E00",dry="#E69F00",semiarid="#0072B2",wet="#009E73",monsoon="#CC79A7")
png(file.path(here,"fig/mri_stability.png"), width=1100, height=460, res=120)
par(mgp=c(2.1,0.6,0),tcl=-0.3,las=1,mar=c(4,4.4,2.6,1),fg=muted,col.axis=ink,col.lab=ink)
plot(NA,xlim=c(0,Tend),ylim=c(0,0.43),xlab="day",ylab=expression(theta[1]~"(top layer, m"^3*" m"^-3*")"),
     bty="n",main="MRI multirate is stable across dry->monsoon scenarios (top-layer soil moisture)")
abline(h=c(0.12,0.428),col=muted,lty=3)
for(sn in names(traj)) lines(daily, traj[[sn]], lwd=1.8, col=cols[sn])
legend("topright", legend=sprintf("%s (%.0f mm/yr, err %.0e)", names(scen),
       sapply(scen,sum), sapply(res,function(z)z$err)), col=cols[names(scen)], lwd=2, bty="n", cex=0.75)
dev.off(); cat("\nwrote fig/mri_stability.png\n")
