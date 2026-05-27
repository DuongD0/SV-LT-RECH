#!/usr/bin/env Rscript
# calibrate_against_stochvol.R
#
# Fit plain SV to a CSV of returns via stochvol::svsample and emit a
# posterior-summary CSV that MATLAB will read back.
#
# Usage:
#   Rscript calibrate_against_stochvol.R <returns_csv> <summary_csv> [draws]
#
# Input CSV columns:  date (YYYY-MM-DD), return  (numeric, in %)
# Output CSV columns: parameter, mean, sd, q025, q975

suppressMessages({
  library(stochvol)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: Rscript calibrate_against_stochvol.R <returns_csv> <summary_csv> [draws]")
}

returns_path <- args[[1L]]
summary_path <- args[[2L]]
draws        <- if (length(args) >= 3L) as.integer(args[[3L]]) else 20000L
burnin       <- max(2000L, draws %/% 5L)

df <- read.csv(returns_path, stringsAsFactors = FALSE)
stopifnot(all(c("date", "return") %in% names(df)))
y <- df$return

fit <- svsample(
  y         = y,
  draws     = draws,
  burnin    = burnin,
  priormu   = c(0, 10),
  priorphi  = c(20, 1.5),
  priorsigma = 1
)

post <- as.matrix(fit$para)
keep <- c("mu", "phi", "sigma")
post <- post[, keep, drop = FALSE]

summary_df <- data.frame(
  parameter = keep,
  mean      = colMeans(post),
  sd        = apply(post, 2L, sd),
  q025      = apply(post, 2L, quantile, probs = 0.025),
  q975      = apply(post, 2L, quantile, probs = 0.975),
  stringsAsFactors = FALSE
)

write.csv(summary_df, file = summary_path, row.names = FALSE)
cat("Wrote ", summary_path, "\n", sep = "")
