"""
OKG_vcf_to_arrays.py — Convert VCFs to numpy arrays and pandas DataFrames
for the LAI HMM pipeline.

Reads thinned reference panels and simulated admixed VCFs, aligns SNP
positions, computes per-population allele frequencies, interpolates
genetic map distances, and saves everything as compressed numpy arrays
and TSV files.

Usage:
    python3 OKG_vcf_to_arrays.py --chrom 22
"""

import sys
import argparse
import numpy as np
import allel
import pandas as pd
from pathlib import Path


def vcf_to_genotype_matrix(vcf_path):
    print(f"Reading {vcf_path}...")
    callset = allel.read_vcf(
        str(vcf_path),
        fields=['variants/CHROM', 'variants/POS', 'variants/REF',
                'variants/ALT', 'calldata/GT', 'samples']
    )
    gt = allel.GenotypeArray(callset['calldata/GT'])
    geno_matrix = gt.to_n_alt().astype(np.int8)

    snp_info = pd.DataFrame({
        'chrom': callset['variants/CHROM'],
        'pos': callset['variants/POS'],
        'ref': callset['variants/REF'],
        'alt': callset['variants/ALT'][:, 0],
    })

    sample_ids = callset['samples']
    print(f"  {geno_matrix.shape[0]} SNPs x {geno_matrix.shape[1]} samples")
    return geno_matrix, snp_info, sample_ids


def compute_allele_freq_table(geno_matrix, sample_ids, sample_pop_map, populations):
    freq_table = pd.DataFrame(index=range(geno_matrix.shape[0]))
    for pop in populations:
        col_idx = [i for i, s in enumerate(sample_ids) if sample_pop_map.get(s) == pop]
        if len(col_idx) == 0:
            print(f"  WARNING: no samples for {pop}")
            freq_table[pop] = np.nan
            continue
        subset = geno_matrix[:, col_idx]
        alt_sum = subset.sum(axis=1).astype(float) + 0.5  # Laplace smoothing
        total = (2 * len(col_idx)) + 1.0
        freq_table[pop] = alt_sum / total
        print(f"  {pop}: {len(col_idx)} samples, mean freq = {freq_table[pop].mean():.4f}")
    return freq_table


def interpolate_genetic_map(snp_positions, map_path):
    data = np.loadtxt(str(map_path), usecols=(2, 3))  # cM, bp
    map_cm = data[:, 0]
    map_bp = data[:, 1].astype(int)
    cm_positions = np.interp(snp_positions, map_bp, map_cm)
    genetic_dist_morgans = np.diff(cm_positions) / 100.0
    genetic_dist_morgans = np.maximum(genetic_dist_morgans, 1e-10)
    print(f"  Genetic span: {cm_positions[0]:.2f} - {cm_positions[-1]:.2f} cM")
    print(f"  Median inter-SNP distance: {np.median(genetic_dist_morgans)*100:.6f} cM")
    return cm_positions, genetic_dist_morgans


def load_sample_pop_map(panel_path):
    df = pd.read_csv(panel_path, sep='\t', header=None, names=['sample', 'pop'])
    return dict(zip(df['sample'], df['pop']))


def process_chromosome(chrom, populations, base):
    ref_vcf     = base / f"haptools_input/reference_panels_thinned/chr{chrom}_unadmixed_snps.vcf.gz"
    query_vcf   = base / f"haptools_sim/sim_admixed_chr{chrom}.vcf.gz"
    sample_info = base / "haptools_sim/sample_info.tsv"
    genetic_map = base / f"haptools_sim/genetic_maps/chr{chrom}.map"
    output_dir  = base / "hmm_input"
    output_dir.mkdir(exist_ok=True)

    for f in [ref_vcf, query_vcf, sample_info, genetic_map]:
        if not f.exists():
            print(f"ERROR: {f} not found")
            sys.exit(1)

    # 1. Reference panel
    print("=" * 60)
    print(f"1. Processing reference panel (chr{chrom})")
    print("=" * 60)
    ref_geno, ref_snp_info, ref_samples = vcf_to_genotype_matrix(ref_vcf)
    sample_pop_map = load_sample_pop_map(sample_info)
    freq_table = compute_allele_freq_table(ref_geno, ref_samples, sample_pop_map, populations)

    # 2. Query genotypes
    print(f"\n2. Processing simulated admixed genotypes (chr{chrom})")
    query_geno, query_snp_info, query_samples = vcf_to_genotype_matrix(query_vcf)

    # 3. Align positions
    print(f"\n3. Aligning SNP positions")
    ref_pos = ref_snp_info['pos'].values
    query_pos = query_snp_info['pos'].values
    shared_pos = np.intersect1d(ref_pos, query_pos)
    ref_mask = np.isin(ref_pos, shared_pos)
    query_mask = np.isin(query_pos, shared_pos)

    positions = ref_pos[ref_mask]
    ref_geno_aligned = ref_geno[ref_mask, :]
    query_geno_aligned = query_geno[query_mask, :]
    freq_table_aligned = freq_table.loc[ref_mask].reset_index(drop=True)
    snp_info_aligned = ref_snp_info.loc[ref_mask].reset_index(drop=True)
    print(f"  Shared SNPs: {len(positions)}")

    # 4. Genetic map
    print(f"\n4. Interpolating genetic map")
    cm_positions, genetic_dist = interpolate_genetic_map(positions, genetic_map)

    # 5. Save
    print(f"\n5. Saving outputs")
    npz_path = output_dir / f"chr{chrom}_hmm_data.npz"
    np.savez_compressed(
        str(npz_path),
        positions=positions,
        query_genotypes=query_geno_aligned,
        ref_genotypes=ref_geno_aligned,
        cm_positions=cm_positions,
        genetic_dist_morgans=genetic_dist,
    )
    print(f"  Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    freq_path = output_dir / f"chr{chrom}_allele_freqs.tsv"
    freq_out = snp_info_aligned.copy()
    for pop in populations:
        freq_out[f'freq_{pop}'] = freq_table_aligned[pop].values
    freq_out.to_csv(freq_path, sep='\t', index=False)
    print(f"  Saved: {freq_path}")

    # Sample lists (only written once)
    for name, ids in [("query_samples.txt", query_samples), ("ref_samples.txt", ref_samples)]:
        path = output_dir / name
        if not path.exists():
            pd.Series(ids).to_csv(path, index=False, header=False)
            print(f"  Saved: {path}")

    # Sanity check
    print(f"\nSanity check (chr{chrom}):")
    freq_diff = np.abs(freq_table_aligned[populations[0]] - freq_table_aligned[populations[1]])
    print(f"  |{populations[0]} - {populations[1]}| mean={freq_diff.mean():.4f}, max={freq_diff.max():.4f}")
    print(f"  SNPs with |diff| > 0.3: {(freq_diff > 0.3).sum()} ({(freq_diff > 0.3).mean()*100:.1f}%)")
    print(f"  SNPs with |diff| > 0.5: {(freq_diff > 0.5).sum()} ({(freq_diff > 0.5).mean()*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VCFs to HMM-ready arrays")
    parser.add_argument("--chrom", required=True, help="Chromosome number (1-22)")
    parser.add_argument("--pops", nargs='+', default=['YRI', 'CEU'],
                        help="Population codes for ancestry states")
    args = parser.parse_args()

    process_chromosome(args.chrom, args.pops, Path("."))
