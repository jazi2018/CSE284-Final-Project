#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_reference_labels(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            rows.append((row[0], row[1]))
    return rows


def write_reference_labels(path: str, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["donor_id", "ancestry"])
        writer.writerows(rows)


def read_ref_panel(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append((row[0], row[1]))
    return rows


def write_ref_panel(path: str, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def subset_reference_panel(
    input_npy: str,
    output_npy: str,
    max_ref_samples: int | None,
    max_snps: int | None
) -> tuple[int, int]:
    panel = np.load(input_npy, mmap_mode="r")

    if panel.ndim != 2:
        raise ValueError("reference_panel.npy must be 2D.")

    n_haps, n_snps_total = panel.shape
    if n_haps % 2 != 0:
        raise ValueError("Expected an even number of haplotype rows in reference_panel.npy.")

    n_ref_samples = n_haps // 2
    keep_ref = n_ref_samples if max_ref_samples is None else min(max_ref_samples, n_ref_samples)
    keep_snps = n_snps_total if max_snps is None else min(max_snps, n_snps_total)

    # reference_panel.npy layout:
    # first n_ref rows are hap0, next n_ref rows are hap1
    keep_rows = list(range(keep_ref)) + list(range(n_ref_samples, n_ref_samples + keep_ref))

    sub = np.asarray(panel[keep_rows, :keep_snps], dtype=np.int8)
    np.save(output_npy, sub)
    return keep_ref, keep_snps

def subset_reference_labels(
    input_tsv: str,
    output_tsv: str,
    keep_ref_samples: int
) -> None:
    rows = read_reference_labels(input_tsv)
    expected = 2 * keep_ref_samples
    if len(rows) < expected:
        raise ValueError(
            f"reference_labels.tsv has only {len(rows)} rows, need at least {expected}."
        )
    write_reference_labels(output_tsv, rows[:expected])


def subset_ref_panel_tsv(
    input_tsv: str,
    output_tsv: str,
    keep_ref_samples: int
) -> None:
    rows = read_ref_panel(input_tsv)
    if len(rows) < keep_ref_samples:
        raise ValueError(
            f"ref_panel.tsv has only {len(rows)} rows, need at least {keep_ref_samples}."
        )
    write_ref_panel(output_tsv, rows[:keep_ref_samples])


def copy_text_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        fout.write(fin.read())


def parse_vcf_header_and_records(vcf_path: str):
    meta = []
    header = None
    records = []

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("##"):
                meta.append(line)
            elif line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
            else:
                records.append(line.rstrip("\n").split("\t"))

    if header is None:
        raise ValueError(f"Could not find VCF header in {vcf_path}")

    return meta, header, records


def subset_vcf(input_vcf, output_vcf, max_samples=None, max_snps=None):
    keep_samples = None
    snp_count = 0

    with open(input_vcf) as fin, open(output_vcf, "w") as fout:
        for line in fin:

            if line.startswith("##"):
                fout.write(line)
                continue

            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                fixed = parts[:9]
                samples = parts[9:]

                if max_samples is None:
                    keep_samples = len(samples)
                else:
                    keep_samples = min(max_samples, len(samples))

                fout.write("\t".join(fixed + samples[:keep_samples]) + "\n")
                continue

            if max_snps is not None and snp_count >= max_snps:
                break

            parts = line.strip().split("\t")
            fixed = parts[:9]
            sample_fields = parts[9:9 + keep_samples]

            fout.write("\t".join(fixed + sample_fields) + "\n")
            snp_count += 1

    return keep_samples, snp_count

def subset_genetic_map(
    input_map: str,
    output_map: str,
    max_snps: int | None
) -> int:
    rows = []
    with open(input_map, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    keep_snps = len(rows) if max_snps is None else min(max_snps, len(rows))
    rows = rows[:keep_snps]

    with open(output_map, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

    return keep_snps


def subset_truth(
    input_truth_gz: str,
    output_truth_gz: str,
    kept_sample_ids: set[str],
    max_snps: int | None
) -> int:
    kept_rows = 0
    allowed_marker_ids = None

    if max_snps is not None:
        allowed_marker_ids = {f"rs{i}" for i in range(1, max_snps + 1)}

    with gzip.open(input_truth_gz, "rt", newline="") as fin, gzip.open(output_truth_gz, "wt", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=["sample_id", "haplotype", "chrom", "pos", "marker_id", "truth_ancestry"],
            delimiter="\t"
        )
        writer.writeheader()

        for row in reader:
            if row["sample_id"] not in kept_sample_ids:
                continue
            if allowed_marker_ids is not None and row["marker_id"] not in allowed_marker_ids:
                continue
            writer.writerow(row)
            kept_rows += 1

    return kept_rows


def read_vcf_sample_ids(vcf_path: str) -> List[str]:
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise ValueError(f"No #CHROM header found in {vcf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a smaller benchmark dataset from an existing benchmark input directory."
    )
    parser.add_argument("--input-dir", required=True, help="Existing benchmark input directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for subset benchmark inputs.")
    parser.add_argument(
        "--max-ref-samples",
        type=int,
        default=None,
        help="Maximum number of reference samples to keep."
    )
    parser.add_argument(
        "--max-query-samples",
        type=int,
        default=None,
        help="Maximum number of query/study samples to keep."
    )
    parser.add_argument(
        "--max-snps",
        type=int,
        default=None,
        help="Maximum number of SNPs to keep (keeps the first max_snps markers)."
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "reference_panel.npy",
        "reference_labels.tsv",
        "ref_panel.tsv",
        "reference.vcf",
        "study.vcf",
        "genetic_map.tsv",
        "ancestry_index.tsv",
        "truth.tsv.gz",
    ]
    missing = [f for f in required_files if not (in_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in input directory: {missing}")

    # 1) reference panel + labels + ref panel
    keep_ref, keep_snps_panel = subset_reference_panel(
        input_npy=str(in_dir / "reference_panel.npy"),
        output_npy=str(out_dir / "reference_panel.npy"),
        max_ref_samples=args.max_ref_samples,
        max_snps=args.max_snps
    )

    subset_reference_labels(
        input_tsv=str(in_dir / "reference_labels.tsv"),
        output_tsv=str(out_dir / "reference_labels.tsv"),
        keep_ref_samples=keep_ref
    )

    subset_ref_panel_tsv(
        input_tsv=str(in_dir / "ref_panel.tsv"),
        output_tsv=str(out_dir / "ref_panel.tsv"),
        keep_ref_samples=keep_ref
    )

    # ancestry index does not change
    copy_text_file(str(in_dir / "ancestry_index.tsv"), str(out_dir / "ancestry_index.tsv"))

    # 2) reference VCF
    keep_ref_vcf, keep_snps_ref = subset_vcf(
        input_vcf=str(in_dir / "reference.vcf"),
        output_vcf=str(out_dir / "reference.vcf"),
        max_samples=keep_ref,
        max_snps=args.max_snps
    )

    # 3) study VCF
    keep_query_vcf, keep_snps_study = subset_vcf(
        input_vcf=str(in_dir / "study.vcf"),
        output_vcf=str(out_dir / "study.vcf"),
        max_samples=args.max_query_samples,
        max_snps=args.max_snps
    )

    if keep_snps_ref != keep_snps_study:
        raise ValueError("Reference and study VCF ended up with different SNP counts.")

    keep_snps = keep_snps_ref

    # 4) genetic map
    keep_snps_map = subset_genetic_map(
        input_map=str(in_dir / "genetic_map.tsv"),
        output_map=str(out_dir / "genetic_map.tsv"),
        max_snps=keep_snps
    )

    if keep_snps_map != keep_snps:
        raise ValueError("Genetic map SNP count does not match VCF SNP count.")

    # 5) truth
    kept_query_sample_ids = set(read_vcf_sample_ids(str(out_dir / "study.vcf")))
    kept_truth_rows = subset_truth(
        input_truth_gz=str(in_dir / "truth.tsv.gz"),
        output_truth_gz=str(out_dir / "truth.tsv.gz"),
        kept_sample_ids=kept_query_sample_ids,
        max_snps=keep_snps
    )

    print("Done.")
    print(f"Input benchmark dir:  {in_dir}")
    print(f"Output benchmark dir: {out_dir}")
    print(f"Reference samples kept: {keep_ref_vcf}")
    print(f"Query samples kept:     {keep_query_vcf}")
    print(f"SNPs kept:              {keep_snps}")
    print(f"Truth rows kept:        {kept_truth_rows}")


if __name__ == "__main__":
    main()