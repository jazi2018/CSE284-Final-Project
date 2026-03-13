#!/usr/bin/env python3

"""
Build benchmark inputs for:
1) your BetterStatesLaihmm method
2) FLARE

from the class files:
- chr21_hmm_data.npz
- ref_samples.txt
- query_samples.txt
- igsr_samples.tsv
- sim_admixed_chr21.bp

Outputs:
- reference_panel.npy
- reference_labels.tsv
- ref_panel.tsv
- reference.vcf
- study.vcf
- genetic_map.tsv
- ancestry_index.tsv
- truth.tsv.gz

Assumptions
-----------
1. The .npz arrays have shapes like:
   - ref_hap0:   (n_snps, n_ref_samples)
   - ref_hap1:   (n_snps, n_ref_samples)
   - query_hap0: (n_snps, n_query_samples)
   - query_hap1: (n_snps, n_query_samples)
   - positions:  (n_snps,)
   - cm_positions: (n_snps,)

2. The .bp file has the observed block format:
      Sample_1_1
      YRI    21    23297046    19.654723
      YRI    21    30845664    30.007421
      CEU    21    41309823    47.305031
      ...
      Sample_1_2
      YRI    21    15855286    2.470394
      CEU    21    25243871    22.057584
      ...

   where:
   - Sample_1_1 means sample Sample_1, haplotype 1
   - Sample_1_2 means sample Sample_1, haplotype 2
   - each ancestry row gives the tract ancestry up to END_BP
   - the next tract begins at previous_end + 1

3. For FLARE:
   - reference.vcf contains the real phased reference haplotypes per sample
   - study.vcf contains the real phased query haplotypes per sample
   - ref_panel.tsv maps each reference sample to ancestry name
   - genetic_map.tsv uses the real bp and cM positions

4. For BetterStatesLaihmm:
   - reference_panel.npy stacks reference haplotypes into shape
     (2 * n_ref_samples, n_snps)
   - reference_labels.tsv duplicates each sample ancestry twice,
     once for hap0 and once for hap1
"""

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Basic file readers
# ---------------------------------------------------------------------

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sniff_delimiter(header_line: str) -> str:
    return "\t" if "\t" in header_line else ","


def normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def read_tsv_like_dicts(path: str) -> Tuple[List[str], List[dict], str]:
    with open(path, "r", newline="") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"Empty file: {path}")
        delim = sniff_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read header from {path}")
        return reader.fieldnames, rows, delim


# ---------------------------------------------------------------------
# IGSR metadata parsing
# ---------------------------------------------------------------------

def build_sample_to_population_code(igsr_path: str) -> Dict[str, str]:
    fieldnames, rows, _ = read_tsv_like_dicts(igsr_path)
    norm_to_orig = {normalize_colname(c): c for c in fieldnames}

    sample_col = None
    pop_col = None

    for cand in ["sample_name", "sample"]:
        if cand in norm_to_orig:
            sample_col = norm_to_orig[cand]
            break

    for cand in ["population_code", "population"]:
        if cand in norm_to_orig:
            pop_col = norm_to_orig[cand]
            break

    if sample_col is None or pop_col is None:
        raise ValueError(
            "Could not find sample/population columns in IGSR metadata. "
            f"Found columns: {fieldnames}"
        )

    out = {}
    for row in rows:
        sample = row[sample_col].strip()
        pop = row[pop_col].strip()
        if sample:
            out[sample] = pop

    return out


# ---------------------------------------------------------------------
# NPZ loading and validation
# ---------------------------------------------------------------------

def load_npz_data(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as data:
        required = [
            "positions",
            "ref_hap0",
            "ref_hap1",
            "query_hap0",
            "query_hap1",
            "cm_positions",
        ]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(f"Missing required arrays in {npz_path}: {missing}")

        positions = np.asarray(data["positions"])
        cm_positions = np.asarray(data["cm_positions"])
        ref_hap0 = np.asarray(data["ref_hap0"])
        ref_hap1 = np.asarray(data["ref_hap1"])
        query_hap0 = np.asarray(data["query_hap0"])
        query_hap1 = np.asarray(data["query_hap1"])

    # Convert from (n_snps, n_samples) to (n_samples, n_snps)
    ref_hap0 = ref_hap0.T
    ref_hap1 = ref_hap1.T
    query_hap0 = query_hap0.T
    query_hap1 = query_hap1.T

    n_snps = positions.shape[0]

    expected_shapes = [
        ("ref_hap0", ref_hap0.shape[1]),
        ("ref_hap1", ref_hap1.shape[1]),
        ("query_hap0", query_hap0.shape[1]),
        ("query_hap1", query_hap1.shape[1]),
        ("cm_positions", cm_positions.shape[0]),
    ]
    bad = [(name, size) for name, size in expected_shapes if size != n_snps]
    if bad:
        raise ValueError(
            f"After transposing haplotypes, SNP dimension mismatch with positions. "
            f"positions has length {n_snps}; mismatches: {bad}"
        )

    for name, arr in [
        ("ref_hap0", ref_hap0),
        ("ref_hap1", ref_hap1),
        ("query_hap0", query_hap0),
        ("query_hap1", query_hap1),
    ]:
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, [0, 1])):
            raise ValueError(
                f"{name} contains values outside {{0,1}}. "
                f"Observed unique values (first 20): {uniq[:20]}"
            )

    return positions, cm_positions, ref_hap0, ref_hap1, query_hap0, query_hap1


# ---------------------------------------------------------------------
# Writing outputs
# ---------------------------------------------------------------------

def write_reference_panel_npy(
    path: Path,
    ref_hap0: np.ndarray,
    ref_hap1: np.ndarray
) -> np.ndarray:
    panel = np.vstack([ref_hap0, ref_hap1]).astype(np.int8)
    np.save(path, panel)
    return panel


def write_reference_labels_tsv(
    path: Path,
    ref_samples: Sequence[str],
    sample_to_pop: Dict[str, str]
) -> Dict[str, int]:
    ancestries_in_order = []

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["donor_id", "ancestry"])

        for sample in ref_samples:
            if sample not in sample_to_pop:
                raise ValueError(f"Reference sample {sample} not found in IGSR metadata.")
            pop = sample_to_pop[sample]

            writer.writerow([f"{sample}_hap0", pop])
            writer.writerow([f"{sample}_hap1", pop])

            ancestries_in_order.extend([pop, pop])

    uniq = []
    seen = set()
    for anc in ancestries_in_order:
        if anc not in seen:
            uniq.append(anc)
            seen.add(anc)

    return {anc: i for i, anc in enumerate(uniq)}


def write_ancestry_index_tsv(path: Path, ancestry_to_index: Dict[str, int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ancestry_name", "ancestry_index"])
        for anc, idx in ancestry_to_index.items():
            writer.writerow([anc, idx])


def write_ref_panel_tsv(
    path: Path,
    ref_samples: Sequence[str],
    sample_to_pop: Dict[str, str]
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for sample in ref_samples:
            if sample not in sample_to_pop:
                raise ValueError(f"Reference sample {sample} not found in IGSR metadata.")
            writer.writerow([sample, sample_to_pop[sample]])


def write_genetic_map_tsv(
    path: Path,
    chrom: str,
    positions: np.ndarray,
    cm_positions: np.ndarray
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for i, (bp, cm) in enumerate(zip(positions, cm_positions), start=1):
            writer.writerow([chrom, f"rs{i}", f"{float(cm):.8f}", int(bp)])


def write_vcf(
    path: Path,
    chrom: str,
    positions: np.ndarray,
    sample_ids: Sequence[str],
    hap0: np.ndarray,
    hap1: np.ndarray
) -> None:
    """
    hap0/hap1 shapes: (n_samples, n_snps), values in {0,1}
    """
    n_samples, n_snps = hap0.shape
    if hap1.shape != (n_samples, n_snps):
        raise ValueError("hap0 and hap1 shapes do not match for VCF writing.")

    with open(path, "w", newline="") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        f.write("\t".join(sample_ids))
        f.write("\n")

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
                f"{int(hap0[sample_idx, snp_idx])}|{int(hap1[sample_idx, snp_idx])}"
                for sample_idx in range(n_samples)
            ]
            row.extend(gts)
            f.write("\t".join(row))
            f.write("\n")


# ---------------------------------------------------------------------
# .bp truth parsing (for the observed block format)
# ---------------------------------------------------------------------

def parse_bp_blocks(
    bp_path: str,
    query_samples: Sequence[str],
    chrom_expected: Optional[str] = None
) -> List[Tuple[str, int, str, int, int]]:
    """
    Parse a .bp file of the observed format:

        Sample_1_1
        YRI    21    23297046    19.654723
        YRI    21    30845664    30.007421
        CEU    21    41309823    47.305031
        ...
        Sample_1_2
        YRI    21    15855286    2.470394
        CEU    21    25243871    22.057584
        ...

    Returns a list of tuples:
        (sample_id, haplotype, ancestry_name, start_bp, end_bp)

    Conventions assumed:
    - block header like "Sample_1_1" means sample_id="Sample_1", haplotype=1
    - block header like "Sample_1_2" means sample_id="Sample_1", haplotype=2
    - each tract row gives ancestry up to an END_BP
    - the next tract starts at previous_end + 1
    - first tract starts at bp 1
    """
    valid_query_samples = set(query_samples)
    segments = []

    current_sample_id = None
    current_haplotype = None
    current_start_bp = 1

    with open(bp_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = re.split(r"\s+", line)

            # Block header line, e.g. Sample_1_1
            if len(fields) == 1:
                block_name = fields[0]

                m = re.fullmatch(r"(.+?)_(1|2)", block_name)
                if not m:
                    raise ValueError(
                        f"Could not parse .bp block header '{block_name}'. "
                        "Expected something like Sample_1_1 or Sample_1_2."
                    )

                current_sample_id = m.group(1)
                current_haplotype = int(m.group(2))
                current_start_bp = 1

                if current_sample_id not in valid_query_samples:
                    raise ValueError(
                        f".bp block sample '{current_sample_id}' not found in query_samples.txt"
                    )

                continue

            # Tract line, e.g. YRI 21 23297046 19.654723
            if len(fields) >= 4:
                if current_sample_id is None or current_haplotype is None:
                    raise ValueError(
                        "Encountered tract row before any sample/haplotype block header in .bp file."
                    )

                ancestry_name = fields[0]
                chrom = fields[1]
                end_bp = int(fields[2])

                if chrom_expected is not None and str(chrom) != str(chrom_expected):
                    raise ValueError(
                        f"Unexpected chromosome in .bp file: got {chrom}, expected {chrom_expected}"
                    )

                if end_bp < current_start_bp:
                    raise ValueError(
                        f".bp tract end {end_bp} is before current start {current_start_bp} "
                        f"for {current_sample_id} haplotype {current_haplotype}"
                    )

                segments.append(
                    (current_sample_id, current_haplotype, ancestry_name, current_start_bp, end_bp)
                )

                current_start_bp = end_bp + 1
                continue

            raise ValueError(f"Unrecognized line in .bp file: {line}")

    return segments


def truth_from_bp_segments(
    segments: Sequence[Tuple[str, int, str, int, int]],
    query_samples: Sequence[str],
    positions: np.ndarray,
    ancestry_to_index: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert parsed .bp segments into truth arrays of shape:
        (n_query_samples, n_snps)
    for haplotype 1 and haplotype 2.
    """
    n_query = len(query_samples)
    n_snps = len(positions)

    truth_h1 = np.full((n_query, n_snps), -1, dtype=int)
    truth_h2 = np.full((n_query, n_snps), -1, dtype=int)

    sample_to_idx = {s: i for i, s in enumerate(query_samples)}

    for sample_id, hap, ancestry_name, start_bp, end_bp in segments:
        if ancestry_name not in ancestry_to_index:
            raise ValueError(
                f"Ancestry '{ancestry_name}' from .bp file not found in ancestry_to_index. "
                f"Known ancestries: {list(ancestry_to_index.keys())}"
            )

        anc = ancestry_to_index[ancestry_name]
        i = sample_to_idx[sample_id]
        mask = (positions >= start_bp) & (positions <= end_bp)

        if hap == 1:
            truth_h1[i, mask] = anc
        elif hap == 2:
            truth_h2[i, mask] = anc
        else:
            raise ValueError(f"Unexpected haplotype value {hap}; expected 1 or 2.")

    return truth_h1, truth_h2


def write_truth_tsv_gz(
    path: Path,
    chrom: str,
    positions: np.ndarray,
    query_samples: Sequence[str],
    truth_h1: np.ndarray,
    truth_h2: np.ndarray
) -> None:
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "sample_id", "haplotype", "chrom", "pos", "marker_id", "truth_ancestry"
        ])

        n_samples, n_snps = truth_h1.shape
        for i, sample_id in enumerate(query_samples):
            for snp_idx in range(n_snps):
                pos = int(positions[snp_idx])
                marker_id = f"rs{snp_idx + 1}"

                if truth_h1[i, snp_idx] != -1:
                    writer.writerow([sample_id, 1, chrom, pos, marker_id, int(truth_h1[i, snp_idx])])
                if truth_h2[i, snp_idx] != -1:
                    writer.writerow([sample_id, 2, chrom, pos, marker_id, int(truth_h2[i, snp_idx])])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build benchmark inputs for BetterStatesLaihmm and FLARE from the class chr21 files."
    )
    parser.add_argument("--npz", required=True, help="Path to chr21_hmm_data.npz")
    parser.add_argument("--ref-samples", required=True, help="Path to ref_samples.txt")
    parser.add_argument("--query-samples", required=True, help="Path to query_samples.txt")
    parser.add_argument("--igsr", required=True, help="Path to igsr_samples.tsv")
    parser.add_argument("--bp", required=True, help="Path to sim_admixed_chr21.bp")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--chrom", default="21", help="Chromosome label to use in VCF/map/truth outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading .npz data...")
    positions, cm_positions, ref_hap0, ref_hap1, query_hap0, query_hap1 = load_npz_data(args.npz)

    print("Reading sample lists...")
    ref_samples = read_lines(args.ref_samples)
    query_samples = read_lines(args.query_samples)

    if len(ref_samples) != ref_hap0.shape[0]:
        raise ValueError(
            f"ref_samples.txt has {len(ref_samples)} samples but ref_hap arrays have "
            f"{ref_hap0.shape[0]} samples after transpose."
        )
    if len(query_samples) != query_hap0.shape[0]:
        raise ValueError(
            f"query_samples.txt has {len(query_samples)} samples but query_hap arrays have "
            f"{query_hap0.shape[0]} samples after transpose."
        )

    print("Reading IGSR metadata...")
    sample_to_pop = build_sample_to_population_code(args.igsr)

    # -----------------------------
    # Outputs for your method
    # -----------------------------
    print("Writing reference_panel.npy ...")
    panel = write_reference_panel_npy(outdir / "reference_panel.npy", ref_hap0, ref_hap1)

    print("Writing reference_labels.tsv ...")
    ancestry_to_index = write_reference_labels_tsv(
        outdir / "reference_labels.tsv",
        ref_samples,
        sample_to_pop
    )

    print("Writing ancestry_index.tsv ...")
    write_ancestry_index_tsv(outdir / "ancestry_index.tsv", ancestry_to_index)

    # -----------------------------
    # Outputs for FLARE
    # -----------------------------
    print("Writing ref_panel.tsv ...")
    write_ref_panel_tsv(outdir / "ref_panel.tsv", ref_samples, sample_to_pop)

    print("Writing reference.vcf ...")
    write_vcf(
        outdir / "reference.vcf",
        chrom=args.chrom,
        positions=positions,
        sample_ids=ref_samples,
        hap0=ref_hap0,
        hap1=ref_hap1
    )

    print("Writing study.vcf ...")
    write_vcf(
        outdir / "study.vcf",
        chrom=args.chrom,
        positions=positions,
        sample_ids=query_samples,
        hap0=query_hap0,
        hap1=query_hap1
    )

    print("Writing genetic_map.tsv ...")
    write_genetic_map_tsv(
        outdir / "genetic_map.tsv",
        chrom=args.chrom,
        positions=positions,
        cm_positions=cm_positions
    )

    # -----------------------------
    # Truth from .bp
    # -----------------------------
    print("Parsing .bp truth file ...")
    segments = parse_bp_blocks(
        bp_path=args.bp,
        query_samples=query_samples,
        chrom_expected=args.chrom
    )

    truth_h1, truth_h2 = truth_from_bp_segments(
        segments=segments,
        query_samples=query_samples,
        positions=positions,
        ancestry_to_index=ancestry_to_index
    )

    missing_h1 = int(np.sum(truth_h1 == -1))
    missing_h2 = int(np.sum(truth_h2 == -1))

    if missing_h1 > 0 or missing_h2 > 0:
        print(
            f"WARNING: truth has uncovered SNPs after .bp conversion "
            f"(hap1 missing={missing_h1}, hap2 missing={missing_h2}). "
            "truth.tsv.gz will still be written with only covered rows."
        )

    print("Writing truth.tsv.gz ...")
    write_truth_tsv_gz(
        outdir / "truth.tsv.gz",
        chrom=args.chrom,
        positions=positions,
        query_samples=query_samples,
        truth_h1=truth_h1,
        truth_h2=truth_h2
    )

    print("\nDone.")
    print(f"Output directory: {outdir}")
    print(f"Reference panel shape for your method: {panel.shape}")
    print(f"Reference VCF samples: {len(ref_samples)}")
    print(f"Study VCF samples: {len(query_samples)}")
    print(f"Number of SNPs: {len(positions)}")
    print(f"Ancestries: {ancestry_to_index}")
    print("truth.tsv.gz written: True")


if __name__ == "__main__":
    main()