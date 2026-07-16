#!/usr/bin/env Rscript
# Independent correctness check: reimplement the exact TF24 soil RHS in R and
# integrate with deSolve (lsoda), then compare to odelia's RKCK and RODAS on the
# same 120-day window. Agreement across three independent integrators validates
# the model and the odelia steppers.
source("compile.R"); compile_soil_runner()
suppressMessages(library(deSolve))
rain_v <- read.csv("out/rainfall.csv")$rain_mm

# --- parameters, identical to soil_runner.cpp ---
theta_sat<-0.428; K_sat<-163.0411; a_psi<-1.78e3; n_psi<-6.57; theta_res<-1e-2
depth<-1500; nL<-5; dz<-depth/nL; a_infil<-1; b_infil<-8; theta_wilt<-0.12
psi_max<-1e3; p_drain<-2*n_psi+3; p_ret<--n_psi
root<-c(0.35,0.28,0.20,0.12,0.05); t_pot<-4.0; stiff<-1.0
Kf<-function(th){r<-pmax(th/theta_sat,0); stiff*K_sat*r^p_drain}
psif<-function(th){t<-pmax(th,theta_res); pmin(a_psi*(t/theta_sat)^p_ret/1e6, psi_max)}
smoothstep<-function(s){s<-pmin(pmax(s,0),1); s*s*(3-2*s)}
stress<-function(th) smoothstep((th-theta_res)/(theta_wilt-theta_res))
rain_at<-function(t){d<-floor(t); d<-min(max(d,0),length(rain_v)-1); rain_v[d+1]}

rhs<-function(t,y,parms){
  th<-y[1:nL]; rain<-rain_at(t)
  runoff<-max(0,1-a_infil*(max(th[1],0)/theta_sat)^b_infil)
  infil<-rain*runoff; out<-Kf(th)
  dth<-numeric(nL)
  for(i in 1:nL){win<-if(i==1)infil else out[i-1]
    dth[i]<-(win-out[i]-t_pot*root[i]*stress(th[i]))/dz}
  dx<-sum(psif(th))          # one reader (M=1, w=0.5..1.5 -> use 1.0 here)
  list(c(dth,dx))
}

Tend<-120; y0<-c(rep(0.30*theta_sat,nL),0)
times<-0:Tend
ds<-deSolve::ode(y0,times,rhs,parms=NULL,method="lsoda",rtol=1e-8,atol=1e-10)
ds_th<-ds[,2:(nL+1)]

# odelia with M=1 reader, matched tol
grid<-0:Tend
ok<-soil_bench(rain_v,grid,M=1L,tol_rel=1e-8,tol_abs=1e-10,method="rkck",stiff=1.0,t_pot=4.0)
od<-soil_bench(rain_v,grid,M=1L,tol_rel=1e-8,tol_abs=1e-10,method="rodas",stiff=1.0,t_pot=4.0)

cat("=== independent cross-check (120 d, M=1) ===\n")
cat(sprintf("deSolve lsoda final theta: [%s]\n",paste(sprintf("%.5f",ds_th[nrow(ds_th),]),collapse=", ")))
cat(sprintf("odelia RKCK  final theta: [%s]\n",paste(sprintf("%.5f",ok$theta[nrow(ok$theta),]),collapse=", ")))
cat(sprintf("odelia RODAS final theta: [%s]\n",paste(sprintf("%.5f",od$theta[nrow(od$theta),]),collapse=", ")))
err_rkck <-max(abs(ok$theta-ds_th))
err_rodas<-max(abs(od$theta-ds_th))
cat(sprintf("max|odelia-deSolve| over trajectory:  RKCK=%.2e  RODAS=%.2e\n",err_rkck,err_rodas))
cat(sprintf("agreement: %s\n", if(max(err_rkck,err_rodas)<1e-3) "PASS (all three integrators agree)" else "CHECK"))
