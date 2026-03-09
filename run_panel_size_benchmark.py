#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path


PANEL_SIZES = [25, 50, 75, 99, 1000]


def run_cmd(cmd):
    print("\nRUNNING:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def read_overall_summary(path: Path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def count_available_ceu_yri(full_bench_dir: Path) -> tuple[int, int]:
    ref_panel_path = full_bench_dir / "ref_panel.tsv"
    if not ref_panel_path.exists():
        raise FileNotFoundError(f"Could not find ref_panel.tsv in {full_bench_dir}")

    n_ceu = 0
    n_yri = 0

    with open(ref_panel_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            ancestry = row[1]
            if ancestry == "CEU":
                n_ceu += 1
            elif ancestry == "YRI":
                n_yri += 1

    return n_ceu, n_yri


def main():
    parser = argparse.ArgumentParser(
        description="Run CEU/YRI local ancestry benchmarks across multiple balanced panel sizes."
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory containing the benchmark scripts and flare.jar."
    )
    parser.add_argument(
        "--full-benchmark-dir",
        default="benchmark_chr21",
        help="Full benchmark input directory built from the raw data."
    )
    parser.add_argument(
        "--max-query-samples",
        type=int,
        default=10,
        help="Number of query samples to keep in each subset benchmark."
    )
    parser.add_argument(
        "--max-snps",
        type=int,
        default=20000,
        help="Number of SNPs to keep in each subset benchmark."
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python executable to use."
    )
    parser.add_argument(
        "--r-bin",
        default="Rscript",
        help="Rscript executable to use."
    )
    parser.add_argument(
        "--java-mem-gb",
        type=int,
        default=8,
        help="Java memory to allocate to FLARE."
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=4,
        help="Number of threads for FLARE."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip panel sizes whose overall_summary.tsv already exists."
    )
    parser.add_argument(
        "--aggregate-out",
        default="panel_size_benchmark_summary.tsv",
        help="Output TSV aggregating all panel-size benchmark summaries."
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    full_bench_dir = (project_dir / args.full_benchmark_dir).resolve()
    flare_jar = project_dir / "flare.jar"

    if not full_bench_dir.exists():
        raise FileNotFoundError(f"Full benchmark dir not found: {full_bench_dir}")
    if not flare_jar.exists():
        raise FileNotFoundError(f"FLARE jar not found: {flare_jar}")

    available_ceu, available_yri = count_available_ceu_yri(full_bench_dir)
    max_balanced = min(available_ceu, available_yri)

    print("=" * 70)
    print("AVAILABLE REFERENCE SAMPLES")
    print("=" * 70)
    print(f"CEU available: {available_ceu}")
    print(f"YRI available: {available_yri}")
    print(f"Maximum balanced CEU/YRI panel size per ancestry: {max_balanced}")

    aggregate_rows = []

    for x in PANEL_SIZES:
        if x > max_balanced:
            print("\n" + "=" * 70)
            print(f"SKIPPING PANEL SIZE {x}/{x}")
            print("=" * 70)
            print(
                f"Not enough balanced CEU/YRI samples available. "
                f"Need {x} of each, but only have CEU={available_ceu}, YRI={available_yri}."
            )
            continue

        label = f"ceu{x}_yri{x}"
        bench_dir = project_dir / f"benchmark_chr21_{label}"
        results_dir = project_dir / f"benchmark_results_chr21_{label}"
        scored_dir = project_dir / f"benchmark_scored_chr21_{label}"

        overall_summary_path = scored_dir / "overall_summary.tsv"

        print("\n" + "=" * 70)
        print(f"PANEL SIZE: {x}/{x} CEU/YRI")
        print("=" * 70)

        if args.skip_existing and overall_summary_path.exists():
            print(f"Skipping existing benchmark for {label}")
        else:
            bench_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)
            scored_dir.mkdir(parents=True, exist_ok=True)

            run_cmd([
                args.python_bin,
                str(project_dir / "subset_ceu_yri_benchmark.py"),
                "--input-dir", str(full_bench_dir),
                "--output-dir", str(bench_dir),
                "--n-ceu", str(x),
                "--n-yri", str(x),
                "--max-query-samples", str(args.max_query_samples),
                "--max-snps", str(args.max_snps),
            ])

            run_cmd([
                args.python_bin,
                str(project_dir / "run_my_method.py"),
                "--reference-panel-npy", str(bench_dir / "reference_panel.npy"),
                "--labels-tsv", str(bench_dir / "reference_labels.tsv"),
                "--ancestry-index-tsv", str(bench_dir / "ancestry_index.tsv"),
                "--study-vcf", str(bench_dir / "study.vcf"),
                "--out-tsv-gz", str(results_dir / f"my_method_{label}.tsv.gz"),
                "--recomb-prob", "0.01",
                "--admixture-prob", "0.001",
                "--error-rate", "0.01",
            ])

            run_cmd([
                args.python_bin,
                str(project_dir / "run_flare.py"),
                "--flare-jar", str(flare_jar),
                "--ref-vcf", str(bench_dir / "reference.vcf"),
                "--ref-panel", str(bench_dir / "ref_panel.tsv"),
                "--study-vcf", str(bench_dir / "study.vcf"),
                "--map", str(bench_dir / "genetic_map.tsv"),
                "--out-prefix", str(results_dir / f"flare_run_{label}"),
                "--parsed-out-tsv-gz", str(results_dir / f"flare_{label}.tsv.gz"),
                "--ancestry-index-tsv", str(bench_dir / "ancestry_index.tsv"),
                "--java-mem-gb", str(args.java_mem_gb),
                "--nthreads", str(args.nthreads),
                "--seed", "1",
                "--probs",
                "--extra-args", "min-mac=1", "min-maf=0",
            ])

            run_cmd([
                args.python_bin,
                str(project_dir / "score_methods.py"),
                "--truth-tsv-gz", str(bench_dir / "truth.tsv.gz"),
                "--prediction-files",
                str(results_dir / f"my_method_{label}.tsv.gz"),
                str(results_dir / f"flare_{label}.tsv.gz"),
                "--outdir", str(scored_dir),
            ])

            run_cmd([
                args.r_bin,
                str(project_dir / "plot_benchmark.R"),
                str(scored_dir),
            ])

        if not overall_summary_path.exists():
            raise FileNotFoundError(f"Missing expected summary file: {overall_summary_path}")

        rows = read_overall_summary(overall_summary_path)
        for row in rows:
            row["panel_size_per_ancestry"] = x
            row["total_reference_samples"] = 2 * x
            row["benchmark_label"] = label
            row["benchmark_dir"] = str(bench_dir)
            row["results_dir"] = str(results_dir)
            row["scored_dir"] = str(scored_dir)
            aggregate_rows.append(row)

    aggregate_out = (project_dir / args.aggregate_out).resolve()
    if not aggregate_rows:
        raise ValueError("No aggregate rows collected. No panel sizes were run successfully.")

    fieldnames = [
        "benchmark_label",
        "panel_size_per_ancestry",
        "total_reference_samples",
        "method",
        "prediction_file",
        "hap_n_rows",
        "hap_accuracy",
        "hap_r",
        "hap_r2",
        "hap_mae",
        "dose_n_rows",
        "dose_accuracy",
        "dose_r",
        "dose_r2",
        "dose_mae",
        "benchmark_dir",
        "results_dir",
        "scored_dir",
    ]

    with open(aggregate_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Aggregate summary written to: {aggregate_out}")


if __name__ == "__main__":
    main()