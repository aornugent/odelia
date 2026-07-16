#!/usr/bin/env Rscript
for (d in c("out","fig")) dir.create(d, showWarnings = FALSE)
# Semi-arid Australian daily rainfall generator.
#
# Chain-dependent (two-state Markov) occurrence + gamma wet-day intensities, with
# summer-dominant seasonality (monsoon-influenced central/western rangelands, e.g.
# Alice Springs / western NSW: ~250-300 mm/yr, >85% dry days, heavy-tailed storms,
# multi-week dry spells). Southern hemisphere: wet season peaks mid-January.
#
# Output: a piecewise-constant daily rainfall flux (mm/day). The daily steps are
# realistic (rain is delivered in daily gauge totals) and are themselves a stress
# for an adaptive ODE solver -- each day boundary is a kink in the forcing.

set.seed(24)                       # reproducible
n_years <- 3L
n_days  <- 365L * n_years
doy     <- ((seq_len(n_days) - 1L) %% 365L)                # 0..364, day 0 = Jan 1

# Seasonal factor: 1 at peak wet (mid-Jan, doy~15), low in winter (mid-Jul).
# cos peaks at doy=15; scaled into [0,1].
season <- (1 + cos(2 * pi * (doy - 15) / 365)) / 2         # 1 summer .. 0 winter

# Occurrence: Markov chain with seasonally modulated transition probs.
# Dry->wet low; wet->wet gives persistence (storm clusters). Winter suppressed.
p01 <- 0.010 + 0.095 * season^1.5    # P(wet tomorrow | dry today)
p11 <- 0.15  + 0.38  * season^1.5    # P(wet tomorrow | wet today)

wet <- logical(n_days)
wet[1] <- FALSE
for (t in 2:n_days) {
  p <- if (wet[t - 1]) p11[t] else p01[t]
  wet[t] <- runif(1) < p
}

# Wet-day amount: gamma, heavy-tailed (shape < 1), larger scale in summer
# (convective storms). shape*scale = mean wet-day depth.
shape <- 0.60
scale <- 4 + 11 * season           # mm; summer storms heavier
rain  <- numeric(n_days)
rain[wet] <- rgamma(sum(wet), shape = shape, scale = scale[wet])
rain <- round(rain, 2)
rain[rain < 0.1] <- 0              # gauge threshold
wet  <- rain > 0

# ---- dry-spell lengths (a key stressor: long depletion toward the residual bound)
rle_dry <- rle(!wet)
dry_runs <- rle_dry$lengths[rle_dry$values]

# ---- summary
ann <- tapply(rain, rep(seq_len(n_years), each = 365L), sum)
cat("=== semi-arid rainfall: summary ===\n")
cat(sprintf("days=%d  years=%d\n", n_days, n_years))
cat(sprintf("annual totals (mm): %s  (mean %.0f)\n",
            paste(sprintf("%.0f", ann), collapse=", "), mean(ann)))
cat(sprintf("wet days: %d (%.1f%%)\n", sum(wet), 100*mean(wet)))
cat(sprintf("max daily (mm): %.1f   95th pct of wet days: %.1f\n",
            max(rain), quantile(rain[wet], 0.95)))
cat(sprintf("longest dry spell: %d days   mean dry spell: %.1f days\n",
            max(dry_runs), mean(dry_runs)))
cat(sprintf("number of >20mm storm days: %d\n", sum(rain > 20)))

# ---- save
df <- data.frame(day = seq_len(n_days) - 1L, doy = doy, rain_mm = rain)
outdir <- "out"
write.csv(df, file.path(outdir, "rainfall.csv"), row.names = FALSE)
cat("wrote", file.path(outdir, "rainfall.csv"), "\n")

# ---- plot
png("fig/rainfall.png", width=1100, height=420)
op <- par(mar=c(4,4,2,1))
plot(df$day, df$rain_mm, type="h", col="#2b6cb0", lwd=1.2,
     xlab="day", ylab="rainfall (mm/day)",
     main=sprintf("Semi-arid daily rainfall (seed 24): %.0f mm/yr, %.1f%% wet days, longest dry spell %d d",
                  mean(ann), 100*mean(wet), max(dry_runs)))
abline(v = seq(0, n_days, 365), col="grey80", lty=2)
par(op); dev.off()
cat("wrote fig/rainfall.png\n")
