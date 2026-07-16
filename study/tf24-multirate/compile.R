# Compile soil_runner.cpp against the installed odelia headers + shared library,
# exactly the way odelia's own tests build standalone runners (helper-load-odelia.R).
for (d in c("out", "fig")) dir.create(d, showWarnings = FALSE)

compile_soil_runner <- function(verbose = FALSE, rebuild = FALSE) {
  suppressMessages(library(odelia))
  inc <- system.file("include", package = "odelia")
  so  <- file.path(system.file("libs", package = "odelia"), "odelia.so")
  stopifnot(dir.exists(inc), file.exists(so))
  Sys.setenv(PKG_CPPFLAGS = paste0("-I", inc))
  Sys.setenv(PKG_LIBS = shQuote(normalizePath(so)))
  Rcpp::sourceCpp("soil_runner.cpp",
                  rebuild = rebuild, verbose = verbose)
  invisible(TRUE)
}
