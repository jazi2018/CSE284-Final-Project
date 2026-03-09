#!/usr/bin/env python3

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


Key = Tuple[str, int, str, int, str]  # sample_id, haplotype, chrom, pos, marker_id
DoseKey = Tuple[str, str, int, str]   # sample_id, chrom, pos, marker_id


def open_auto(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return open(path, mode, newline="")


def read_truth(path: str) -> Dict[Key, int]:
    truth = {}
    with open_auto(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (
                row["sample_id"],
                int(row["haplotype"]),
                row["chrom"],
                int(row["pos"]),
                row["marker_id"]
            )
            truth[key] = int(row["truth_ancestry"])
    return truth


def read_predictions(path: str) -> Tuple[str, Dict[Key, int]]:
    preds = {}
    method_name = None
    with open_auto(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if method_name is None:
                method_name = row["method"]

            key = (
                row["sample_id"],
                int(row["haplotype"]),
                row["chrom"],
                int(row["pos"]),
                row["marker_id"]
            )
            preds[key] = int(row["pred_ancestry"])
    if method_name is None:
        raise ValueError(f"No rows found in prediction file: {path}")
    return method_name, preds


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize_haplotype_level(truth: Dict[Key, int], preds: Dict[Key, int]):
    shared_keys = sorted(set(truth.keys()) & set(preds.keys()))
    y_true = np.array([truth[k] for k in shared_keys], dtype=float)
    y_pred = np.array([preds[k] for k in shared_keys], dtype=float)

    acc = float(np.mean(y_true == y_pred)) if len(shared_keys) else np.nan
    r = safe_corr(y_true, y_pred)
    r2 = r ** 2 if not np.isnan(r) else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(shared_keys) else np.nan

    return {
        "n_rows": len(shared_keys),
        "accuracy": acc,
        "pearson_r": r,
        "r2": r2,
        "mae": mae
    }


def collapse_to_dosage(d: Dict[Key, int]) -> Dict[DoseKey, int]:
    grouped = defaultdict(dict)
    for (sample_id, hap, chrom, pos, marker_id), anc in d.items():
        grouped[(sample_id, chrom, pos, marker_id)][hap] = anc

    out = {}
    for dose_key, hapvals in grouped.items():
        if 1 in hapvals and 2 in hapvals:
            out[dose_key] = int(hapvals[1]) + int(hapvals[2])
    return out


def summarize_dosage_level(truth: Dict[Key, int], preds: Dict[Key, int]):
    truth_dose = collapse_to_dosage(truth)
    pred_dose = collapse_to_dosage(preds)

    shared_keys = sorted(set(truth_dose.keys()) & set(pred_dose.keys()))
    y_true = np.array([truth_dose[k] for k in shared_keys], dtype=float)
    y_pred = np.array([pred_dose[k] for k in shared_keys], dtype=float)

    acc = float(np.mean(y_true == y_pred)) if len(shared_keys) else np.nan
    r = safe_corr(y_true, y_pred)
    r2 = r ** 2 if not np.isnan(r) else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(shared_keys) else np.nan

    return {
        "n_rows": len(shared_keys),
        "accuracy": acc,
        "pearson_r": r,
        "r2": r2,
        "mae": mae
    }


def per_sample_summary(truth: Dict[Key, int], preds: Dict[Key, int]):
    by_sample_truth = defaultdict(dict)
    by_sample_pred = defaultdict(dict)

    for k, v in truth.items():
        by_sample_truth[k[0]][k] = v
    for k, v in preds.items():
        by_sample_pred[k[0]][k] = v

    rows = []
    for sample_id in sorted(set(by_sample_truth) & set(by_sample_pred)):
        t = by_sample_truth[sample_id]
        p = by_sample_pred[sample_id]
        hap = summarize_haplotype_level(t, p)
        dose = summarize_dosage_level(t, p)
        rows.append({
            "sample_id": sample_id,
            "hap_n_rows": hap["n_rows"],
            "hap_accuracy": hap["accuracy"],
            "hap_r": hap["pearson_r"],
            "hap_r2": hap["r2"],
            "hap_mae": hap["mae"],
            "dose_n_rows": dose["n_rows"],
            "dose_accuracy": dose["accuracy"],
            "dose_r": dose["pearson_r"],
            "dose_r2": dose["r2"],
            "dose_mae": dose["mae"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score local ancestry predictions against truth.")
    parser.add_argument("--truth-tsv-gz", required=True, help="Truth file from simulate_admixed.py")
    parser.add_argument("--prediction-files", nargs="+", required=True, help="One or more standardized prediction files.")
    parser.add_argument("--outdir", required=True, help="Directory for summary outputs.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    truth = read_truth(args.truth_tsv_gz)

    overall_rows = []

    for pred_file in args.prediction_files:
        method, preds = read_predictions(pred_file)

        hap = summarize_haplotype_level(truth, preds)
        dose = summarize_dosage_level(truth, preds)
        overall_rows.append({
            "method": method,
            "prediction_file": pred_file,
            "hap_n_rows": hap["n_rows"],
            "hap_accuracy": hap["accuracy"],
            "hap_r": hap["pearson_r"],
            "hap_r2": hap["r2"],
            "hap_mae": hap["mae"],
            "dose_n_rows": dose["n_rows"],
            "dose_accuracy": dose["accuracy"],
            "dose_r": dose["pearson_r"],
            "dose_r2": dose["r2"],
            "dose_mae": dose["mae"],
        })

        sample_rows = per_sample_summary(truth, preds)
        per_sample_path = outdir / f"{method}.per_sample_summary.tsv"
        with open(per_sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(sample_rows)

    overall_path = outdir / "overall_summary.tsv"
    with open(overall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(overall_rows)

    print(f"Wrote {overall_path}")


if __name__ == "__main__":
    main()