#!/usr/bin/env bash
###############################################################################
# OKG_process_chrom.sh — Thin reference, simulate admixture, convert to arrays.
# Usage: bash OKG_process_chrom.sh <chrom_number>
###############################################################################

set -euo pipefail

# Activate the project venv (needed when run via nohup)
source .venv/bin/activate

CHR="$1"

if [ -z "$CHR" ]; then
    echo "Usage: bash OKG_process_chrom.sh <chrom_number>"
    exit 1
fi

echo ""
echo "############################################################"
echo "# CHROMOSOME ${CHR} — $(date)"
echo "############################################################"
echo ""

REF_THINNED="./haptools_input/reference_panels_thinned/chr${CHR}_unadmixed_snps.vcf.gz"
SIM_VCF="./haptools_sim/sim_admixed_chr${CHR}.vcf.gz"
NPZ_OUT="./hmm_input/chr${CHR}_hmm_data.npz"

# ── Step 1: Thin reference panel ──────────────────────────────
echo "[chr${CHR}] Step 1: Thinning reference panel..."
bash ./OKG_thin_reference.sh "${CHR}"

VARIANT_COUNT=$(bcftools index -n "$REF_THINNED")
echo "[chr${CHR}] Thinned reference has ${VARIANT_COUNT} variants."

# ── Step 2: Simulate (skip if output already exists) ──────────
if [ -f "$SIM_VCF" ]; then
    echo "[chr${CHR}] Simulation output already exists, skipping haptools."
else
    echo "[chr${CHR}] Running haptools simgenotype..."
    haptools simgenotype \
        --model ./haptools_sim/afr_eur_admixture.dat \
        --mapdir ./haptools_sim/genetic_maps/ \
        --chroms "${CHR}" \
        --ref_vcf "$REF_THINNED" \
        --sample_info ./haptools_sim/sample_info.tsv \
        --pop_field \
        --seed 42 \
        --out "$SIM_VCF"
    echo "[chr${CHR}] Simulation done."
fi

# ── Step 3: Convert to arrays (skip if output already exists) ─
if [ -f "$NPZ_OUT" ]; then
    echo "[chr${CHR}] NPZ already exists, skipping conversion."
else
    echo "[chr${CHR}] Converting to HMM arrays..."
    python3 ./OKG_vcf_to_arrays.py --chrom "${CHR}"
    echo "[chr${CHR}] Conversion done."
fi

echo "[chr${CHR}] COMPLETE — $(date)"
