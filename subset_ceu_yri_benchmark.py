#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def read_ref_panel(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append((row[0], row[1]))
    return rows


def read_reference_labels(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            rows.append((row[0], row[1]))
    return rows


def read_ancestry_index(path):
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            out[row[0]] = int(row[1])
    return out


def write_ancestry_index(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ancestry_name", "ancestry_index"])
        writer.writerow(["CEU", 0])
        writer.writerow(["YRI", 1])


def subset_reference_panel(input_npy, output_npy, selected_ref_indices, max_snps):
    panel = np.load(input_npy, mmap_mode="r")
    n_haps, n_snps_total = panel.shape
    n_ref = n_haps // 2
    keep_snps = min(max_snps, n_snps_total) if max_snps is not None else n_snps_total

    keep_rows = selected_ref_indices + [n_ref + i for i in selected_ref_indices]
    sub = np.asarray(panel[keep_rows, :keep_snps], dtype=np.int8)
    np.save(output_npy, sub)


def write_reference_labels(output_tsv, selected_ref_rows):
    with open(output_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["donor_id", "ancestry"])
        for sample, anc in selected_ref_rows:
            writer.writerow([f"{sample}_hap0", anc])
            writer.writerow([f"{sample}_hap1", anc])


def write_ref_panel(output_tsv, selected_ref_rows):
    with open(output_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(selected_ref_rows)


def subset_vcf_by_selected_samples(input_vcf, output_vcf, selected_sample_names, max_snps):
    selected_sample_names = list(selected_sample_names)
    keep_snps = 0

    with open(input_vcf) as fin, open(output_vcf, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue

            if line.startswith("#CHROM"):
                parts = line.rstrip("\n").split("\t")
                fixed = parts[:9]
                samples = parts[9:]
                sample_to_idx = {s: i for i, s in enumerate(samples)}
                selected_indices = [sample_to_idx[s] for s in selected_sample_names]
                fout.write("\t".join(fixed + selected_sample_names) + "\n")
                continue

            if max_snps is not None and keep_snps >= max_snps:
                break

            parts = line.rstrip("\n").split("\t")
            fixed = parts[:9]
            sample_fields = parts[9:]
            kept = [sample_fields[i] for i in selected_indices]
            fout.write("\t".join(fixed + kept) + "\n")
            keep_snps += 1

    return keep_snps


def subset_vcf_first_n_samples(input_vcf, output_vcf, max_samples, max_snps):
    keep_snps = 0
    kept_sample_names = None

    with open(input_vcf) as fin, open(output_vcf, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue

            if line.startswith("#CHROM"):
                parts = line.rstrip("\n").split("\t")
                fixed = parts[:9]
                samples = parts[9:]
                keep_n = min(max_samples, len(samples)) if max_samples is not None else len(samples)
                kept_sample_names = samples[:keep_n]
                fout.write("\t".join(fixed + kept_sample_names) + "\n")
                continue

            if max_snps is not None and keep_snps >= max_snps:
                break

            parts = line.rstrip("\n").split("\t")
            fixed = parts[:9]
            sample_fields = parts[9:9 + len(kept_sample_names)]
            fout.write("\t".join(fixed + sample_fields) + "\n")
            keep_snps += 1

    return kept_sample_names, keep_snps


def subset_genetic_map(input_map, output_map, max_snps):
    rows = []
    with open(input_map, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    keep = min(max_snps, len(rows)) if max_snps is not None else len(rows)
    with open(output_map, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows[:keep])


def recode_truth(input_truth_gz, output_truth_gz, kept_query_samples, max_snps, old_ancestry_index):
    old_inv = {v: k for k, v in old_ancestry_index.items()}
    new_map = {"CEU": 0, "YRI": 1}
    allowed_markers = {f"rs{i}" for i in range(1, max_snps + 1)} if max_snps is not None else None

    with gzip.open(input_truth_gz, "rt", newline="") as fin, gzip.open(output_truth_gz, "wt", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=["sample_id", "haplotype", "chrom", "pos", "marker_id", "truth_ancestry"],
            delimiter="\t"
        )
        writer.writeheader()

        for row in reader:
            if row["sample_id"] not in kept_query_samples:
                continue
            if allowed_markers is not None and row["marker_id"] not in allowed_markers:
                continue

            anc_name = old_inv[int(row["truth_ancestry"])]
            if anc_name not in new_map:
                continue

            row["truth_ancestry"] = str(new_map[anc_name])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ceu", type=int, required=True)
    parser.add_argument("--n-yri", type=int, required=True)
    parser.add_argument("--max-query-samples", type=int, required=True)
    parser.add_argument("--max-snps", type=int, required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_panel_rows = read_ref_panel(in_dir / "ref_panel.tsv")
    old_ancestry_index = read_ancestry_index(in_dir / "ancestry_index.tsv")

    ceu_rows = [(i, row) for i, row in enumerate(ref_panel_rows) if row[1] == "CEU"]
    yri_rows = [(i, row) for i, row in enumerate(ref_panel_rows) if row[1] == "YRI"]

    if len(ceu_rows) < args.n_ceu or len(yri_rows) < args.n_yri:
        raise ValueError("Not enough CEU or YRI reference samples available.")

    selected = ceu_rows[:args.n_ceu] + yri_rows[:args.n_yri]
    selected_ref_indices = [i for i, _ in selected]
    selected_ref_rows = [row for _, row in selected]
    selected_ref_names = [row[0] for row in selected_ref_rows]

    subset_reference_panel(
        in_dir / "reference_panel.npy",
        out_dir / "reference_panel.npy",
        selected_ref_indices,
        args.max_snps
    )

    write_reference_labels(out_dir / "reference_labels.tsv", selected_ref_rows)
    write_ref_panel(out_dir / "ref_panel.tsv", selected_ref_rows)
    write_ancestry_index(out_dir / "ancestry_index.tsv")

    subset_vcf_by_selected_samples(
        in_dir / "reference.vcf",
        out_dir / "reference.vcf",
        selected_ref_names,
        args.max_snps
    )

    kept_query_samples, keep_snps = subset_vcf_first_n_samples(
        in_dir / "study.vcf",
        out_dir / "study.vcf",
        args.max_query_samples,
        args.max_snps
    )

    subset_genetic_map(
        in_dir / "genetic_map.tsv",
        out_dir / "genetic_map.tsv",
        keep_snps
    )

    recode_truth(
        in_dir / "truth.tsv.gz",
        out_dir / "truth.tsv.gz",
        set(kept_query_samples),
        keep_snps,
        old_ancestry_index
    )

    print("Done.")
    print(f"CEU kept: {args.n_ceu}")
    print(f"YRI kept: {args.n_yri}")
    print(f"Query samples kept: {len(kept_query_samples)}")
    print(f"SNPs kept: {keep_snps}")


if __name__ == "__main__":
    main()