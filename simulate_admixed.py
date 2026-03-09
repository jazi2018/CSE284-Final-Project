#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_labels(labels_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Read donor IDs and ancestry labels from a two-column TSV:
        donor_id    ancestry
    """
    donor_ids = []
    ancestries = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 2:
            raise ValueError("labels file must have at least two columns: donor_id, ancestry")

        for row in reader:
            donor_ids.append(row[0])
            ancestries.append(row[1])

    return donor_ids, np.array(ancestries, dtype=object)


def build_default_positions(n_snps: int, step_bp: int = 1000) -> np.ndarray:
    return np.arange(1, n_snps * step_bp + 1, step_bp, dtype=int)


def build_default_cm(n_snps: int, step_cm: float = 0.001) -> np.ndarray:
    return np.arange(0, n_snps * step_cm, step_cm, dtype=float)


def write_plink_map(path: Path, chrom: str, positions: np.ndarray, cm_positions: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for i, (bp, cm) in enumerate(zip(positions, cm_positions), start=1):
            writer.writerow([chrom, f"rs{i}", f"{cm:.6f}", int(bp)])


def write_ref_panel(path: Path, donor_ids: List[str], ancestries: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for donor_id, anc in zip(donor_ids, ancestries):
            writer.writerow([donor_id, anc])


def write_vcf(
    path: Path,
    chrom: str,
    positions: np.ndarray,
    sample_ids: List[str],
    hap1_matrix: np.ndarray,
    hap2_matrix: np.ndarray
) -> None:
    """
    hap1_matrix, hap2_matrix: shape (n_samples, n_snps), values in {0,1}
    """
    with open(path, "w", newline="") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        f.write("\t".join(sample_ids))
        f.write("\n")

        n_samples, n_snps = hap1_matrix.shape
        for snp_idx in range(n_snps):
            row = [
                chrom,
                str(int(positions[snp_idx])),
                f"rs{snp_idx + 1}",
                "A",
                "G",
                ".",
                "PASS",
                ".",
                "GT",
            ]
            gts = [
                f"{int(hap1_matrix[sample_idx, snp_idx])}|{int(hap2_matrix[sample_idx, snp_idx])}"
                for sample_idx in range(n_samples)
            ]
            row.extend(gts)
            f.write("\t".join(row))
            f.write("\n")


def write_truth(
    path: Path,
    chrom: str,
    positions: np.ndarray,
    sample_ids: List[str],
    truth_h1: np.ndarray,
    truth_h2: np.ndarray
) -> None:
    """
    truth_h1, truth_h2: shape (n_samples, n_snps), ancestry coded as integers
    """
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "sample_id", "haplotype", "chrom", "pos", "marker_id", "truth_ancestry"
        ])

        n_samples, n_snps = truth_h1.shape
        for i, sample_id in enumerate(sample_ids):
            for snp_idx in range(n_snps):
                marker_id = f"rs{snp_idx + 1}"
                pos = int(positions[snp_idx])

                writer.writerow([sample_id, 1, chrom, pos, marker_id, int(truth_h1[i, snp_idx])])
                writer.writerow([sample_id, 2, chrom, pos, marker_id, int(truth_h2[i, snp_idx])])


def simulate_haplotype_path(
    donor_indices_by_ancestry: Dict[str, np.ndarray],
    ancestry_to_index: Dict[str, int],
    n_snps: int,
    recomb_prob: float,
    admixture_prob: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a donor path and ancestry path for one haplotype.
    ancestry changes with admixture_prob; donor-within-ancestry changes with recomb_prob.
    """
    ancestry_names = list(donor_indices_by_ancestry.keys())

    current_anc_name = ancestry_names[rng.integers(len(ancestry_names))]
    current_donor = int(rng.choice(donor_indices_by_ancestry[current_anc_name]))

    donor_path = np.zeros(n_snps, dtype=int)
    anc_path = np.zeros(n_snps, dtype=int)

    for t in range(n_snps):
        if t > 0:
            u = rng.random()
            if u < admixture_prob:
                other_ancs = [a for a in ancestry_names if a != current_anc_name]
                current_anc_name = other_ancs[rng.integers(len(other_ancs))]
                current_donor = int(rng.choice(donor_indices_by_ancestry[current_anc_name]))
            elif u < admixture_prob + recomb_prob:
                same_pool = donor_indices_by_ancestry[current_anc_name]
                if len(same_pool) > 1:
                    choices = same_pool[same_pool != current_donor]
                    if len(choices) > 0:
                        current_donor = int(rng.choice(choices))

        donor_path[t] = current_donor
        anc_path[t] = ancestry_to_index[current_anc_name]

    return donor_path, anc_path


def main():
    parser = argparse.ArgumentParser(description="Simulate phased admixed haplotypes and write benchmark inputs.")
    parser.add_argument("--reference-panel-npy", required=True, help="NumPy .npy file of shape (n_donors, n_snps), values in {0,1}.")
    parser.add_argument("--labels-tsv", required=True, help="TSV with columns donor_id and ancestry.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--num-admixed", type=int, default=100, help="Number of simulated admixed individuals.")
    parser.add_argument("--chrom", default="1", help="Chromosome label for VCF/map output.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")
    parser.add_argument("--recomb-prob", type=float, default=0.01, help="Within-ancestry donor-switch probability.")
    parser.add_argument("--admixture-prob", type=float, default=0.001, help="Between-ancestry switch probability.")
    parser.add_argument("--bp-step", type=int, default=1000, help="Base-pair spacing if no positions file is provided.")
    parser.add_argument("--cm-step", type=float, default=0.001, help="cM spacing for the synthetic genetic map.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    reference_panel = np.load(args.reference_panel_npy)
    if reference_panel.ndim != 2:
        raise ValueError("reference_panel must be 2D with shape (n_donors, n_snps).")

    donor_ids, ancestry_labels = read_labels(args.labels_tsv)
    if reference_panel.shape[0] != len(donor_ids):
        raise ValueError("Number of rows in reference panel must match number of donors in labels file.")

    n_donors, n_snps = reference_panel.shape
    positions = build_default_positions(n_snps, step_bp=args.bp_step)
    cm_positions = build_default_cm(n_snps, step_cm=args.cm_step)

    unique_ancestries = list(dict.fromkeys(ancestry_labels.tolist()))
    ancestry_to_index = {anc: i for i, anc in enumerate(unique_ancestries)}

    donor_indices_by_ancestry = {
        anc: np.where(ancestry_labels == anc)[0]
        for anc in unique_ancestries
    }

    for anc, idxs in donor_indices_by_ancestry.items():
        if len(idxs) == 0:
            raise ValueError(f"No donors found for ancestry {anc}.")

    # reference VCF: treat each donor as a phased diploid homozygote from one haplotype row
    ref_h1 = reference_panel.copy()
    ref_h2 = reference_panel.copy()

    # study VCF + truth
    study_ids = [f"ADMIXED_{i+1}" for i in range(args.num_admixed)]
    study_h1 = np.zeros((args.num_admixed, n_snps), dtype=int)
    study_h2 = np.zeros((args.num_admixed, n_snps), dtype=int)
    truth_h1 = np.zeros((args.num_admixed, n_snps), dtype=int)
    truth_h2 = np.zeros((args.num_admixed, n_snps), dtype=int)

    for i in range(args.num_admixed):
        donor_path1, anc_path1 = simulate_haplotype_path(
            donor_indices_by_ancestry=donor_indices_by_ancestry,
            ancestry_to_index=ancestry_to_index,
            n_snps=n_snps,
            recomb_prob=args.recomb_prob,
            admixture_prob=args.admixture_prob,
            rng=rng
        )
        donor_path2, anc_path2 = simulate_haplotype_path(
            donor_indices_by_ancestry=donor_indices_by_ancestry,
            ancestry_to_index=ancestry_to_index,
            n_snps=n_snps,
            recomb_prob=args.recomb_prob,
            admixture_prob=args.admixture_prob,
            rng=rng
        )

        study_h1[i, :] = reference_panel[donor_path1, np.arange(n_snps)]
        study_h2[i, :] = reference_panel[donor_path2, np.arange(n_snps)]
        truth_h1[i, :] = anc_path1
        truth_h2[i, :] = anc_path2

    write_vcf(
        outdir / "reference.vcf",
        chrom=args.chrom,
        positions=positions,
        sample_ids=donor_ids,
        hap1_matrix=ref_h1,
        hap2_matrix=ref_h2
    )

    write_vcf(
        outdir / "study.vcf",
        chrom=args.chrom,
        positions=positions,
        sample_ids=study_ids,
        hap1_matrix=study_h1,
        hap2_matrix=study_h2
    )

    write_ref_panel(outdir / "ref_panel.tsv", donor_ids, ancestry_labels)
    write_plink_map(outdir / "genetic_map.tsv", args.chrom, positions, cm_positions)
    write_truth(outdir / "truth.tsv.gz", args.chrom, positions, study_ids, truth_h1, truth_h2)

    with open(outdir / "ancestry_index.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ancestry_name", "ancestry_index"])
        for anc, idx in ancestry_to_index.items():
            writer.writerow([anc, idx])

    print("Done.")
    print(f"Output directory: {outdir}")
    print(f"Ancestries and indices: {ancestry_to_index}")


if __name__ == "__main__":
    main()