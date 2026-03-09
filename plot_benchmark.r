#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("Usage: Rscript plot_benchmark.R <scored_dir>")
}

scored_dir <- args[1]

overall_path <- file.path(scored_dir, "overall_summary.tsv")
if (!file.exists(overall_path)) {
  stop("Could not find: ", overall_path)
}

overall <- read.delim(overall_path, stringsAsFactors = FALSE, check.names = FALSE)

per_sample_files <- list.files(
  scored_dir,
  pattern = "\\.per_sample_summary\\.tsv$",
  full.names = TRUE
)

dir.create(file.path(scored_dir, "figures"), showWarnings = FALSE)
fig_dir <- file.path(scored_dir, "figures")

# -----------------------------
# Overall barplots
# -----------------------------
png(file.path(fig_dir, "overall_haplotype_r2.png"), width = 1000, height = 700)
barplot(
  height = overall$hap_r2,
  names.arg = overall$method,
  main = "Haplotype-level R-squared by method",
  ylab = expression(R^2),
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "overall_dosage_r2.png"), width = 1000, height = 700)
barplot(
  height = overall$dose_r2,
  names.arg = overall$method,
  main = "Diploid ancestry dosage R-squared by method",
  ylab = expression(R^2),
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "overall_haplotype_accuracy.png"), width = 1000, height = 700)
barplot(
  height = overall$hap_accuracy,
  names.arg = overall$method,
  main = "Haplotype-level accuracy by method",
  ylab = "Accuracy",
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "overall_dosage_accuracy.png"), width = 1000, height = 700)
barplot(
  height = overall$dose_accuracy,
  names.arg = overall$method,
  main = "Diploid ancestry dosage accuracy by method",
  ylab = "Accuracy",
  ylim = c(0, 1)
)
dev.off()

# -----------------------------
# Per-sample boxplots
# -----------------------------
all_per_sample <- do.call(
  rbind,
  lapply(per_sample_files, function(fp) {
    x <- read.delim(fp, stringsAsFactors = FALSE, check.names = FALSE)
    method <- sub("\\.per_sample_summary\\.tsv$", "", basename(fp))
    x$method <- method
    x
  })
)

png(file.path(fig_dir, "per_sample_haplotype_r2_boxplot.png"), width = 1000, height = 700)
boxplot(
  hap_r2 ~ method,
  data = all_per_sample,
  main = "Per-sample haplotype-level R-squared",
  ylab = expression(R^2),
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "per_sample_dosage_r2_boxplot.png"), width = 1000, height = 700)
boxplot(
  dose_r2 ~ method,
  data = all_per_sample,
  main = "Per-sample diploid ancestry dosage R-squared",
  ylab = expression(R^2),
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "per_sample_haplotype_accuracy_boxplot.png"), width = 1000, height = 700)
boxplot(
  hap_accuracy ~ method,
  data = all_per_sample,
  main = "Per-sample haplotype-level accuracy",
  ylab = "Accuracy",
  ylim = c(0, 1)
)
dev.off()

png(file.path(fig_dir, "per_sample_dosage_accuracy_boxplot.png"), width = 1000, height = 700)
boxplot(
  dose_accuracy ~ method,
  data = all_per_sample,
  main = "Per-sample diploid ancestry dosage accuracy",
  ylab = "Accuracy",
  ylim = c(0, 1)
)
dev.off()

cat("Wrote figures to:", fig_dir, "\n")