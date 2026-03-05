#!/usr/bin/env bash
###############################################################################
# OKGDL.sh — Download 1000 Genomes Phase 3 VCFs (22 autosomes + panel file)
#
# Downloads ~15GB of phased genotype VCFs from the EBI FTP mirror.
# Uses wget -c for resume support; safe to re-run if interrupted.
#
# Usage:
#   bash OKGDL.sh
###############################################################################

set -euo pipefail

OUTDIR="./1kg_phase3_vcfs"
BASE="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
SUFFIX=".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

mkdir -p "$OUTDIR"

echo "=== Downloading 1000 Genomes Phase 3 VCFs ==="
echo "    22 autosomes + panel file"
echo "    Destination: ${OUTDIR}"
echo ""

# Download panel file (sample-to-population mapping)
if [ ! -f "${OUTDIR}/integrated_call_samples_v3.20130502.ALL.panel" ]; then
    echo "Downloading panel file..."
    wget -c -P "$OUTDIR" \
        "${BASE}/integrated_call_samples_v3.20130502.ALL.panel"
else
    echo "Panel file already exists, skipping."
fi

# Download per-chromosome VCFs
for chr in {1..22}; do
    FILENAME="ALL.chr${chr}${SUFFIX}"
    echo ""
    echo "[${chr}/22] Chromosome ${chr}"

    if [ -f "${OUTDIR}/${FILENAME}" ]; then
        echo "    Already exists, skipping."
        continue
    fi

    wget -c -P "$OUTDIR" "${BASE}/${FILENAME}"
done

echo ""
echo "=== Download complete ==="
echo ""
ls -lhS "${OUTDIR}"/*.vcf.gz 2>/dev/null | awk '{print "  "$5, $9}'
