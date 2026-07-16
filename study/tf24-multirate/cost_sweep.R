#!/usr/bin/env Rscript
for (d in c("out","fig")) dir.create(d, showWarnings = FALSE)
# Consistent cost-vs-size sweep on the two-way soil+canopy model (same model for
# every curve), full 3-year horizon, matched tol 1e-6. Global RODAS only up to
# N=205 (dense O(N^3) makes larger N impractical -- itself the finding).
suppressMessages(library(odelia))
inc <- system.file("include", package="odelia"); so <- file.path(system.file("libs", package="odelia"), "odelia.so")
Sys.setenv(PKG_CPPFLAGS=paste0("-I",inc)); Sys.setenv(PKG_LIBS=shQuote(normalizePath(so)))
Rcpp::sourceCpp("multirate_runner.cpp")
D<-"."
rain <- read.csv(file.path(D,"out/rainfall.csv"))$rain_mm; Tend<-length(rain); daily<-0:Tend
rows<-list(); add<-function(...) rows[[length(rows)+1]]<<-data.frame(...)
for (M in c(50L,100L,200L,400L,800L)) {
  gk <- global_run(rain, daily, M, 1e-6,1e-8,"rkck",1.0,4.0); add(method="rk45",N=5+M,wall_ms=gk$wall_ms)
  H1 <- 0:Tend
  mr <- multirate_run(rain,H1,M,1e-6,1e-8,"rkck",1.0,4.0,0.02,0.10); add(method="multirate",N=5+M,wall_ms=mr$wall_ms)
  mri<- multirate_run(rain,H1,M,1e-6,1e-8,"rodas",1.0,4.0,0.02,0.10); add(method="multirate_imex",N=5+M,wall_ms=mri$wall_ms)
  if (M<=200L) { gr<-global_run(rain,daily,M,1e-6,1e-8,"rodas",1.0,4.0); add(method="rodas",N=5+M,wall_ms=gr$wall_ms) }
  cat(sprintf("N=%d done\n",5+M))
}
df<-do.call(rbind,rows); write.csv(df, file.path(D,"out/cost_scaling.csv"), row.names=FALSE)
cat("wrote out/cost_scaling.csv\n"); print(df)
