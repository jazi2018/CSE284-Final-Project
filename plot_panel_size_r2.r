#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("Usage: Rscript plot_panel_size_r2.R <aggregate_summary_tsv>")
}

summary_path <- args[1]

if (!file.exists(summary_path)) {
  stop("Could not find: ", summary_path)
}

df <- read.delim(summary_path, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c("panel_size_per_ancestry", "method", "hap_r2", "dose_r2", "hap_accuracy", "dose_accuracy")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df$panel_size_per_ancestry <- as.numeric(df$panel_size_per_ancestry)
df$hap_r2 <- suppressWarnings(as.numeric(df$hap_r2))
df$dose_r2 <- suppressWarnings(as.numeric(df$dose_r2))
df$hap_accuracy <- suppressWarnings(as.numeric(df$hap_accuracy))
df$dose_accuracy <- suppressWarnings(as.numeric(df$dose_accuracy))

out_dir <- file.path(dirname(summary_path), "panel_size_figures")
dir.create(out_dir, showWarnings = FALSE)

plot_metric <- function(data, ycol, ylab, main, filename) {
  png(filename, width = 1000, height = 700)

  methods <- unique(data$method)
  ok_any <- FALSE

  xlim <- range(data$panel_size_per_ancestry, na.rm = TRUE)
  ylim <- c(0, 1)

  plot(
    NA,
    xlim = xlim,
    ylim = ylim,
    log = "x",
    xlab = "Reference panel size per ancestry (X in X/X)",
    ylab = ylab,
    main = main,
    xaxt = "n"
  )

  axis(1, at = sort(unique(data$panel_size_per_ancestry)), labels = sort(unique(data$panel_size_per_ancestry)))

  colors <- c("my_method" = "black", "flare" = "blue")
  pchs <- c("my_method" = 16, "flare" = 17)
  ltys <- c("my_method" = 1, "flare" = 2)

  for (m in methods) {
    d <- data[data$method == m, ]
    d <- d[order(d$panel_size_per_ancestry), ]
    y <- d[[ycol]]
    ok <- is.finite(y)

    if (any(ok)) {
      ok_any <- TRUE
      lines(
        d$panel_size_per_ancestry[ok],
        y[ok],
        type = "b",
        col = ifelse(m %in% names(colors), colors[[m]], "gray40"),
        pch = ifelse(m %in% names(pchs), pchs[[m]], 16),
        lty = ifelse(m %in% names(ltys), ltys[[m]], 1),
        lwd = 2
      )
    }
  }

  if (!ok_any) {
    plot.new()
    text(0.5, 0.5, "No finite values to plot")
  } else {
    legend(
      "bottomright",
      legend = methods,
      col = sapply(methods, function(m) ifelse(m %in% names(colors), colors[[m]], "gray40")),
      pch = sapply(methods, function(m) ifelse(m %in% names(pchs), pchs[[m]], 16)),
      lty = sapply(methods, function(m) ifelse(m %in% names(ltys), ltys[[m]], 1)),
      lwd = 2,
      bty = "n"
    )
  }

  dev.off()
}

plot_metric(
  df,
  ycol = "hap_r2",
  ylab = expression(Haplotype ~ R^2),
  main = "Haplotype-level R-squared vs. reference panel size",
  filename = file.path(out_dir, "haplotype_r2_vs_panel_size.png")
)

plot_metric(
  df,
  ycol = "dose_r2",
  ylab = expression(Dosage ~ R^2),
  main = "Diploid ancestry dosage R-squared vs. reference panel size",
  filename = file.path(out_dir, "dosage_r2_vs_panel_size.png")
)

plot_metric(
  df,
  ycol = "hap_accuracy",
  ylab = "Haplotype accuracy",
  main = "Haplotype accuracy vs. reference panel size",
  filename = file.path(out_dir, "haplotype_accuracy_vs_panel_size.png")
)

plot_metric(
  df,
  ycol = "dose_accuracy",
  ylab = "Dosage accuracy",
  main = "Diploid ancestry dosage accuracy vs. reference panel size",
  filename = file.path(out_dir, "dosage_accuracy_vs_panel_size.png")
)

cat("Wrote panel-size figures to:", out_dir, "\n")