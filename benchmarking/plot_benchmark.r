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

safe_boxplot <- function(formula, data, filename, main, ylab) {
  mf <- model.frame(formula, data = data)
  y <- mf[[1]]
  ok <- is.finite(y)

  png(filename, width = 1000, height = 700)
  if (!any(ok)) {
    plot.new()
    text(0.5, 0.5, "No finite values to plot")
  } else {
    boxplot(
      formula,
      data = data[ok, , drop = FALSE],
      main = main,
      ylab = ylab,
      ylim = c(0, 1)
    )
  }
  dev.off()
}

safe_barplot <- function(values, names_arg, filename, main, ylab) {
  vals <- suppressWarnings(as.numeric(values))
  ok <- is.finite(vals)

  png(filename, width = 1000, height = 700)
  if (!any(ok)) {
    plot.new()
    text(0.5, 0.5, "No finite values to plot")
  } else {
    barplot(
      height = vals[ok],
      names.arg = names_arg[ok],
      main = main,
      ylab = ylab,
      ylim = c(0, 1)
    )
  }
  dev.off()
}

# -----------------------------
# Overall barplots
# -----------------------------
safe_barplot(
  values = overall$hap_r2,
  names_arg = overall$method,
  filename = file.path(fig_dir, "overall_haplotype_r2.png"),
  main = "Haplotype-level R-squared by method",
  ylab = expression(R^2)
)

safe_barplot(
  values = overall$dose_r2,
  names_arg = overall$method,
  filename = file.path(fig_dir, "overall_dosage_r2.png"),
  main = "Diploid ancestry dosage R-squared by method",
  ylab = expression(R^2)
)

safe_barplot(
  values = overall$hap_accuracy,
  names_arg = overall$method,
  filename = file.path(fig_dir, "overall_haplotype_accuracy.png"),
  main = "Haplotype-level accuracy by method",
  ylab = "Accuracy"
)

safe_barplot(
  values = overall$dose_accuracy,
  names_arg = overall$method,
  filename = file.path(fig_dir, "overall_dosage_accuracy.png"),
  main = "Diploid ancestry dosage accuracy by method",
  ylab = "Accuracy"
)

# -----------------------------
# Per-sample boxplots
# -----------------------------
if (length(per_sample_files) == 0) {
  message("No per-sample summary files found; writing placeholder plots.")

  placeholder_files <- c(
    "per_sample_haplotype_r2_boxplot.png",
    "per_sample_dosage_r2_boxplot.png",
    "per_sample_haplotype_accuracy_boxplot.png",
    "per_sample_dosage_accuracy_boxplot.png"
  )

  for (fn in placeholder_files) {
    png(file.path(fig_dir, fn), width = 1000, height = 700)
    plot.new()
    text(0.5, 0.5, "No per-sample summary files found")
    dev.off()
  }
} else {
  all_per_sample <- do.call(
    rbind,
    lapply(per_sample_files, function(fp) {
      x <- read.delim(fp, stringsAsFactors = FALSE, check.names = FALSE)
      method <- sub("\\.per_sample_summary\\.tsv$", "", basename(fp))
      x$method <- method
      x
    })
  )

  safe_boxplot(
    hap_r2 ~ method,
    data = all_per_sample,
    filename = file.path(fig_dir, "per_sample_haplotype_r2_boxplot.png"),
    main = "Per-sample haplotype-level R-squared",
    ylab = expression(R^2)
  )

  safe_boxplot(
    dose_r2 ~ method,
    data = all_per_sample,
    filename = file.path(fig_dir, "per_sample_dosage_r2_boxplot.png"),
    main = "Per-sample diploid ancestry dosage R-squared",
    ylab = expression(R^2)
  )

  safe_boxplot(
    hap_accuracy ~ method,
    data = all_per_sample,
    filename = file.path(fig_dir, "per_sample_haplotype_accuracy_boxplot.png"),
    main = "Per-sample haplotype-level accuracy",
    ylab = "Accuracy"
  )

  safe_boxplot(
    dose_accuracy ~ method,
    data = all_per_sample,
    filename = file.path(fig_dir, "per_sample_dosage_accuracy_boxplot.png"),
    main = "Per-sample diploid ancestry dosage accuracy",
    ylab = "Accuracy"
  )
}

cat("Wrote figures to:", fig_dir, "\n")