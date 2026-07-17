#!/usr/bin/env Rscript
# Phase-B validation: reverse-mode gradients through the multirate MRI scheme, on a
# realistic workflow matrix (rainfall scenarios x trait combinations), plus |theta|
# and M scaling. Robustness = every gradient matches frozen-schedule FD; performance
# = reverse cost O(1) in |theta| and multirate cost sub-linear/cheap in M.
suppressMessages(library(odelia))
inc<-system.file("include",package="odelia"); so<-file.path(system.file("libs",package="odelia"),"odelia.so")
here<-"/home/user/plant-dev/scripts/tf24-multirate"; dir.create(file.path(here,"fig"),showWarnings=FALSE)
Sys.setenv(PKG_CPPFLAGS=paste0("-I",inc," -I",here)); Sys.setenv(PKG_LIBS=shQuote(normalizePath(so)))
Rcpp::sourceCpp(file.path(here,"mri_ad_runner.cpp"))
cat("=== compiled reverse-mode runner ===\n")

# ---- rainfall scenario generator (Markov occurrence + gamma intensity) --------
gen_rain <- function(seed, ndays=365, p01b=0.09, p11b=0.38, shape=0.6, scaleb=11, occ=1) {
  set.seed(seed); doy<-(seq_len(ndays)-1)%%365
  season<-(1+cos(2*pi*(doy-15)/365))/2
  p01<-pmin(0.9,(0.010+p01b*season^1.5)*occ); p11<-pmin(0.95,0.15+p11b*season^1.5)
  wet<-logical(ndays)
  for(t in 2:ndays) wet[t]<-runif(1) < (if(wet[t-1])p11[t] else p01[t])
  scale<-4+scaleb*season; rain<-numeric(ndays)
  rain[wet]<-rgamma(sum(wet),shape=shape,scale=scale[wet]); rain[rain<0.1]<-0
  round(rain,2)
}
scen <- list(
  normal = gen_rain(24),
  dry    = gen_rain(7,  occ=0.45, scaleb=7),
  wet    = gen_rain(11, occ=1.8,  scaleb=16),
  pulsy  = gen_rain(3,  occ=0.5,  shape=0.4, scaleb=30)
)
for(nm in names(scen)) cat(sprintf("scenario %-7s: %4.0f mm/yr, %2.0f%% wet, max %2.0f mm\n",
  nm, sum(scen[[nm]]), 100*mean(scen[[nm]]>0), max(scen[[nm]])))

# ---- trait sampling (realistic ranges), fixed seed -----------------------------
set.seed(101); NT<-12
lo<-c(0.3,4.0,2.0,0.06,0.5); hi<-c(3.0,9.0,6.0,0.18,2.0)
nm_tr<-c("Ksat_mult","n_psi","t_pot","theta_wilt","alpha_scale")
Traits<-matrix(runif(NT*5), NT, 5); for(j in 1:5) Traits[,j]<-lo[j]+(hi[j]-lo[j])*Traits[,j]

macro<-0:365; M<-50L; DELTA<-1e-3   # smooth near-bound floors (declared model option)
run_matrix <- function(delta) {
  allrev<-c(); allfd<-c(); wabs<-0; wabs_where<-""; npass<-0; ntot<-0
  for(sn in names(scen)) for(ti in 1:NT){
    r<-mri_grad(scen[[sn]], Traits[ti,], M, macro, "midpoint", 1e-8,1e-10, 0.02,0.10, 1e-6, TRUE, integer(0), delta)
    ntot<-ntot+1; if(r$norm_err<1) npass<-npass+1        # |rev-fd| < atol + rtol|fd|
    if(r$max_abs_err>wabs){ wabs<-r$max_abs_err; wabs_where<-sprintf("%s/trait%d",sn,ti) }
    allrev<-c(allrev, r$grad_rev); allfd<-c(allfd, r$grad_fd)
  }
  list(allrev=allrev, allfd=allfd, wabs=wabs, where=wabs_where, npass=npass, ntot=ntot)
}
cat(sprintf("\n=== robustness: %d scenarios x %d trait vectors (M=%d, midpoint, H=1) ===\n",
    length(scen), NT, M))
cat("  pass = |grad_rev - grad_FD| < 1e-7 + 1e-4*|grad_FD|  (abs+rel, robust to ~0 components)\n")
hard <- run_matrix(0.0)      # hard positivity clamps
cat(sprintf("HARD clamps  : %d/%d match FD   (worst abs err %.1e at %s)\n", hard$npass, hard$ntot, hard$wabs, hard$where))
sm <- run_matrix(DELTA)      # smooth near-bound floors (declared model option)
cat(sprintf("SMOOTH floors: %d/%d match FD   (worst abs err %.1e at %s)\n", sm$npass, sm$ntot, sm$wabs, sm$where))
allrev<-sm$allrev; allfd<-sm$allfd; worst<-sm$wabs

# ---- |theta| scaling: reverse O(1), FD O(|theta|) ------------------------------
cat("\n=== |theta| scaling (normal scenario, M=50): reverse vs FD wall ===\n")
cat(sprintf("%-8s %12s %12s %10s\n","n_active","reverse_ms","FD_ms","FD/rev"))
th_scale<-data.frame()
for(na in 1:5){ act<-0:(na-1)
  r<-mri_grad(scen$normal, Traits[1,], M, macro, "midpoint",1e-8,1e-10,0.02,0.10,1e-6,TRUE,act,DELTA)
  cat(sprintf("%-8d %12.1f %12.1f %10.2f\n", na, r$wall_rev_ms, r$wall_fd_ms, r$wall_fd_ms/r$wall_rev_ms))
  th_scale<-rbind(th_scale, data.frame(na=na, rev=r$wall_rev_ms, fd=r$wall_fd_ms))
}

# ---- M scaling: forward + reverse wall (reverse cheap in M) ---------------------
cat("\n=== M scaling (normal scenario, 5 traits): forward vs reverse wall ===\n")
cat(sprintf("%-6s %12s %12s %10s %12s\n","M","fwd_ms","reverse_ms","rev/fwd","soil_steps"))
m_scale<-data.frame()
for(Mv in c(10L,50L,200L,800L)){
  r<-mri_grad(scen$normal, Traits[1,], Mv, macro, "midpoint",1e-8,1e-10,0.02,0.10,1e-6,FALSE,integer(0),DELTA)
  cat(sprintf("%-6d %12.1f %12.1f %10.2f %12.0f\n", Mv, r$wall_fwd_ms, r$wall_rev_ms, r$rev_over_fwd, r$soil_steps))
  m_scale<-rbind(m_scale, data.frame(M=Mv, fwd=r$wall_fwd_ms, rev=r$wall_rev_ms))
}

# ---- figures ------------------------------------------------------------------
ink<-"#222"; muted<-"#8a8a8a"
png(file.path(here,"fig/mri_ad_validation.png"), width=1150, height=460, res=120)
par(mfrow=c(1,3), mgp=c(2.1,0.6,0), tcl=-0.3, las=1, mar=c(4,4.2,2.6,1), fg=muted, col.axis=ink, col.lab=ink)
# (a) reverse vs FD correctness
rng<-range(c(allrev,allfd))
plot(allfd, allrev, pch=16, cex=0.5, col=adjustcolor("#0072B2",0.5), xlim=rng, ylim=rng,
     xlab="gradient (frozen-schedule FD)", ylab="gradient (reverse-mode tape)",
     main="(a) reverse = FD across 48x5 gradients"); abline(0,1,col="#D55E00",lwd=1.5)
text(rng[1],rng[2],sprintf("worst abs err\n%.1e",worst),pos=4,cex=0.8,col=ink)
# (b) |theta| scaling
plot(th_scale$na, th_scale$fd, type="b", pch=17, lwd=2, col="#D55E00", ylim=range(0,th_scale$fd,th_scale$rev),
     xlab="# traits differentiated", ylab="wall (ms)", main="(b) reverse O(1) vs FD O(|θ|)")
lines(th_scale$na, th_scale$rev, type="b", pch=16, lwd=2, col="#009E73")
legend("topleft", c("finite difference","reverse-mode"), pch=c(17,16), col=c("#D55E00","#009E73"), lwd=2, bty="n", cex=0.85)
# (c) M scaling
plot(m_scale$M, m_scale$rev, type="b", pch=16, lwd=2, col="#009E73", log="xy",
     ylim=range(m_scale$fwd,m_scale$rev), xlab="canopy block size M", ylab="wall (ms)",
     main="(c) reverse cost vs big-block size M")
lines(m_scale$M, m_scale$fwd, type="b", pch=15, lwd=2, col="#0072B2")
legend("topleft", c("reverse gradient","forward solve"), pch=c(16,15), col=c("#009E73","#0072B2"), lwd=2, bty="n", cex=0.85)
dev.off(); cat("\nwrote fig/mri_ad_validation.png\n")
