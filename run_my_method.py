#!/usr/bin/env python3

import argparse
import csv
import gzip
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from model import BetterStatesLaihmm


def read_labels(labels_path: str) -> Tuple[List[str], np.ndarray]:
    donor_ids = []
    ancestries = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            donor_ids.append(row[0])
            ancestries.append(row[1])
    return donor_ids, np.array(ancestries, dtype=object)


def read_ancestry_index(path: str) -> dict:
    mapping = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            mapping[row[0]] = int(row[1])
    return mapping


def parse_phased_vcf(vcf_path: str) -> Tuple[List[str], str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        sample_ids
        chrom
        positions
        hap1_matrix (n_samples, n_snps)
        hap2_matrix (n_samples, n_snps)
    """
    sample_ids = None
    chrom = None
    positions = []
    hap1_rows = []
    hap2_rows = []

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                sample_ids = fields[9:]
                hap1_rows = [[] for _ in sample_ids]
                hap2_rows = [[] for _ in sample_ids]
                continue

            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            positions.append(pos)

            gts = fields[9:]
            for i, gt_field in enumerate(gts):
                gt = gt_field.split(":")[0]
                a1, a2 = gt.split("|")
                hap1_rows[i].append(int(a1))
                hap2_rows[i].append(int(a2))

    if sample_ids is None:
        raise ValueError("VCF header not found.")

    hap1_matrix = np.array(hap1_rows, dtype=int)
    hap2_matrix = np.array(hap2_rows, dtype=int)
    return sample_ids, chrom, np.array(positions, dtype=int), hap1_matrix, hap2_matrix


def main():
    parser = argparse.ArgumentParser(description="Run BetterStatesLaihmm on simulated study haplotypes.")
    parser.add_argument("--reference-panel-npy", required=True, help="Reference haplotype panel (.npy), shape (n_donors, n_snps).")
    parser.add_argument("--labels-tsv", required=True, help="TSV with donor_id and ancestry.")
    parser.add_argument("--ancestry-index-tsv", required=True, help="TSV with ancestry_name and ancestry_index.")
    parser.add_argument("--study-vcf", required=True, help="Phased study VCF.")
    parser.add_argument("--out-tsv-gz", required=True, help="Output standardized predictions file.")
    parser.add_argument("--recomb-prob", type=float, default=0.01)
    parser.add_argument("--admixture-prob", type=float, default=0.001)
    parser.add_argument("--error-rate", type=float, default=0.01)
    args = parser.parse_args()

    reference_panel = np.load(args.reference_panel_npy)
    _, ancestry_names = read_labels(args.labels_tsv)
    ancestry_index = read_ancestry_index(args.ancestry_index_tsv)

    ancestry_labels_numeric = np.array([ancestry_index[a] for a in ancestry_names], dtype=int)

    sample_ids, chrom, positions, hap1_matrix, hap2_matrix = parse_phased_vcf(args.study_vcf)

    model = BetterStatesLaihmm(
        reference_panel=reference_panel,
        ancestry_labels=ancestry_labels_numeric,
        recombination_prob=args.recomb_prob,
        admixture_prob=args.admixture_prob
    )

    out_path = Path(args.out_tsv_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "method",
            "sample_id",
            "haplotype",
            "chrom",
            "pos",
            "marker_id",
            "pred_ancestry",
            "pred_prob"
        ])

        for sample_idx, sample_id in enumerate(sample_ids):
            pred_h1 = np.asarray(model.predict(hap1_matrix[sample_idx, :], error_rate=args.error_rate), dtype=int)
            pred_h2 = np.asarray(model.predict(hap2_matrix[sample_idx, :], error_rate=args.error_rate), dtype=int)

            for snp_idx, pos in enumerate(positions):
                marker_id = f"rs{snp_idx + 1}"
                writer.writerow([
                    "my_method",
                    sample_id,
                    1,
                    chrom,
                    int(pos),
                    marker_id,
                    int(pred_h1[snp_idx]),
                    ""
                ])
                writer.writerow([
                    "my_method",
                    sample_id,
                    2,
                    chrom,
                    int(pos),
                    marker_id,
                    int(pred_h2[snp_idx]),
                    ""
                ])

    elapsed = time.perf_counter() - start

    runtime_path = out_path.with_suffix("").with_suffix(".runtime.tsv")
    with open(runtime_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["method", "seconds", "n_samples", "n_snps"])
        writer.writerow(["my_method", elapsed, len(sample_ids), len(positions)])

    print(f"Wrote predictions to {out_path}")
    print(f"Wrote runtime to {runtime_path}")


if __name__ == "__main__":
    main()