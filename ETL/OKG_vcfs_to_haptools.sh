#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# OKG_vcfs_to_haptools.sh
# Filter 1000 Genomes Phase 3 VCFs into unadmixed reference panels and
# admixed sample sets for Haptools simulation and LAI.
#
# Prerequisites: bcftools >= 1.19, tabix, OKGDL.sh completed
# Usage: bash OKG_vcfs_to_haptools.sh
###############################################################################

INPUT_DIR="./1kg_phase3_vcfs"
OUTPUT_DIR="./haptools_input"
PANEL_FILE="${INPUT_DIR}/integrated_call_samples_v3.20130502.ALL.panel"
SUFFIX=".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

# AFR: YRI LWK GWD MSL ESN | EUR: CEU GBR FIN IBS TSI
# EAS: CHB JPT CHS CDX KHV | SAS: GIH PJL BEB STU ITU
UNADMIXED_POPS="YRI|LWK|GWD|MSL|ESN|CEU|GBR|FIN|IBS|TSI|CHB|JPT|CHS|CDX|KHV|GIH|PJL|BEB|STU|ITU"
ADMIXED_POPS="MXL|PUR|CLM|PEL|ACB|ASW"
CHROMS=($(seq 1 22))

mkdir -p "${OUTPUT_DIR}"/{reference_panels,admixed,sample_maps,logs}

echo "=== 1000 Genomes → Haptools Preparation ==="

# ── Step 1: Sample lists ─────────────────────────────────────
echo "[Step 1/4] Building sample lists..."
[[ -f "$PANEL_FILE" ]] || { echo "ERROR: Panel file not found. Run OKGDL.sh first."; exit 1; }

grep -E "${UNADMIXED_POPS}" "$PANEL_FILE" | cut -f1 > "${OUTPUT_DIR}/sample_maps/unadmixed_samples.txt"
grep -E "${ADMIXED_POPS}" "$PANEL_FILE"   | cut -f1 > "${OUTPUT_DIR}/sample_maps/admixed_samples.txt"
awk 'NR > 1 {print $1"\t"$2"\t"$3}' "$PANEL_FILE" > "${OUTPUT_DIR}/sample_maps/sample_pop_map.tsv"

for pop in $(echo "$UNADMIXED_POPS" | tr '|' ' '); do
    awk -v p="$pop" '$2 == p {print $1}' "$PANEL_FILE" \
        > "${OUTPUT_DIR}/reference_panels/${pop}_samples.txt"
done

echo "    $(wc -l < "${OUTPUT_DIR}/sample_maps/unadmixed_samples.txt") unadmixed, $(wc -l < "${OUTPUT_DIR}/sample_maps/admixed_samples.txt") admixed"

# ── Step 2: Unadmixed reference panels ──────────────────────
echo "[Step 2/4] Filtering unadmixed reference panels (biallelic SNPs)..."
TOTAL=${#CHROMS[@]}
for i in "${!CHROMS[@]}"; do
    chr=${CHROMS[$i]}; n=$((i + 1))
    IN_VCF="${INPUT_DIR}/ALL.chr${chr}${SUFFIX}"
    OUT_VCF="${OUTPUT_DIR}/reference_panels/chr${chr}_unadmixed_snps.vcf.gz"
    [[ -f "$IN_VCF" ]] || { echo "    [${n}/${TOTAL}] chr${chr} — SKIPPED"; continue; }
    echo -n "    [${n}/${TOTAL}] chr${chr}..."
    bcftools view -S "${OUTPUT_DIR}/sample_maps/unadmixed_samples.txt" \
        -m2 -M2 -v snps --min-ac 1:nref "$IN_VCF" \
        -Oz -o "$OUT_VCF" 2> "${OUTPUT_DIR}/logs/chr${chr}_unadmixed.log"
    tabix -p vcf "$OUT_VCF"
    echo " $(bcftools index -n "$OUT_VCF") variants"
done

# ── Step 3: Admixed individuals ──────────────────────────────
echo "[Step 3/4] Filtering admixed individuals..."
for i in "${!CHROMS[@]}"; do
    chr=${CHROMS[$i]}; n=$((i + 1))
    IN_VCF="${INPUT_DIR}/ALL.chr${chr}${SUFFIX}"
    OUT_VCF="${OUTPUT_DIR}/admixed/chr${chr}_admixed_snps.vcf.gz"
    [[ -f "$IN_VCF" ]] || { echo "    [${n}/${TOTAL}] chr${chr} — SKIPPED"; continue; }
    echo -n "    [${n}/${TOTAL}] chr${chr}..."
    bcftools view -S "${OUTPUT_DIR}/sample_maps/admixed_samples.txt" \
        -m2 -M2 -v snps --min-ac 1:nref "$IN_VCF" \
        -Oz -o "$OUT_VCF" 2> "${OUTPUT_DIR}/logs/chr${chr}_admixed.log"
    tabix -p vcf "$OUT_VCF"
    echo " $(bcftools index -n "$OUT_VCF") variants"
done

# ── Step 4: Summary ──────────────────────────────────────────
echo "[Step 4/4] Done."
echo "Reference panels:"; ls -lhS "${OUTPUT_DIR}/reference_panels/"chr*_unadmixed_snps.vcf.gz 2>/dev/null | awk '{print "  "$5, $9}'
echo "Admixed VCFs:";     ls -lhS "${OUTPUT_DIR}/admixed/"chr*_admixed_snps.vcf.gz 2>/dev/null | awk '{print "  "$5, $9}'
echo "=== Ready for Haptools simgenotype ==="

# ── Also create haptools sample_info.tsv (2-col: sample, pop) ─
SAMPLE_INFO="./haptools_sim/sample_info.tsv"
mkdir -p ./haptools_sim
awk 'NR > 1 {print $1"\t"$2}' "$PANEL_FILE" > "$SAMPLE_INFO"
echo "Created ${SAMPLE_INFO} ($(wc -l < "$SAMPLE_INFO") samples)"
