#!/usr/bin/env bash
###############################################################################
# OKG_thin_reference.sh — Thin a reference panel VCF to cap variant count.
# Prevents OOM during Haptools simulation on machines with ≤32GB RAM.
# Chromosomes under the cap are copied as-is.
#
# Usage: bash OKG_thin_reference.sh <chrom>
###############################################################################

set -euo pipefail

CHR="$1"
MAX_VARIANTS=1500000

INPUT="./haptools_input/reference_panels/chr${CHR}_unadmixed_snps.vcf.gz"
OUTDIR="./haptools_input/reference_panels_thinned"
OUTPUT="${OUTDIR}/chr${CHR}_unadmixed_snps.vcf.gz"

mkdir -p "$OUTDIR"

if [ -f "$OUTPUT" ]; then
    echo "[chr${CHR}] Thinned VCF already exists, skipping."
    exit 0
fi

if [ ! -f "$INPUT" ]; then
    echo "[chr${CHR}] ERROR: ${INPUT} not found"
    exit 1
fi

TOTAL=$(bcftools index -n "$INPUT")
echo "[chr${CHR}] Raw variants: ${TOTAL}"

if [ "$TOTAL" -le "$MAX_VARIANTS" ]; then
    echo "[chr${CHR}] Under cap (${MAX_VARIANTS}), copying as-is."
    cp "$INPUT" "$OUTPUT"
    cp "${INPUT}.tbi" "${OUTPUT}.tbi" 2>/dev/null || tabix -p vcf "$OUTPUT"
else
    # GRCh37 chromosome lengths
    declare -A CHR_LEN=(
        [1]=249250621  [2]=243199373  [3]=198022430  [4]=191154276
        [5]=180915260  [6]=171115067  [7]=159138663  [8]=146364022
        [9]=141213431  [10]=135534747 [11]=135006516 [12]=133851895
        [13]=115169878 [14]=107349540 [15]=102531392 [16]=90354753
        [17]=81195210  [18]=78077248  [19]=59128983  [20]=63025520
        [21]=48129895  [22]=51304566
    )

    SPAN=${CHR_LEN[$CHR]}
    MIN_DIST=$((SPAN / MAX_VARIANTS))

    echo "[chr${CHR}] Thinning: span=${SPAN}bp, min_dist=${MIN_DIST}bp, target ~${MAX_VARIANTS}"

    bcftools query -f '%CHROM\t%POS\n' "$INPUT" | \
        awk -v dist="$MIN_DIST" 'NR==1 || $2 - last >= dist {print; last=$2}' > /tmp/thin_pos_chr${CHR}.txt

    KEPT=$(wc -l < /tmp/thin_pos_chr${CHR}.txt)
    echo "[chr${CHR}] Keeping ${KEPT} variants (from ${TOTAL})"

    bcftools view -T /tmp/thin_pos_chr${CHR}.txt "$INPUT" -Oz -o "$OUTPUT"
    tabix -p vcf "$OUTPUT"

    rm /tmp/thin_pos_chr${CHR}.txt
    echo "[chr${CHR}] Thinned VCF written."
fi
