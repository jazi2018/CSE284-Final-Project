#!/usr/bin/env bash
###############################################################################
# OKG_run_all_chroms.sh — Process all 22 chromosomes in ascending size order.
# Thins reference panels, simulates admixture with Haptools, and converts
# outputs to numpy arrays for the HMM.
#
# Usage:
#   nohup bash OKG_run_all_chroms.sh > pipeline.log 2>&1 &
#   tail -f pipeline.log
###############################################################################

set -euo pipefail

# Ascending order by GRCh37 chromosome size
CHROMS=(21 22 19 20 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1)

TOTAL=${#CHROMS[@]}
DONE=0
FAILED=0
START_TIME=$(date +%s)

echo "============================================================"
echo "LAI PIPELINE (memory-safe) — $(date)"
echo "Processing ${TOTAL} chromosomes (smallest to largest)"
echo "Max variants per chromosome: 1,500,000"
echo "============================================================"

for CHR in "${CHROMS[@]}"; do
    DONE=$((DONE + 1))
    echo ""
    echo "[${DONE}/${TOTAL}] Starting chr${CHR}..."

    if bash ./OKG_process_chrom.sh "${CHR}"; then
        echo "[${DONE}/${TOTAL}] chr${CHR} succeeded."
    else
        echo "[${DONE}/${TOTAL}] chr${CHR} FAILED (exit code $?). Continuing..."
        FAILED=$((FAILED + 1))
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    echo "[Progress] ${DONE}/${TOTAL} done, ${FAILED} failed, elapsed ${HOURS}h${MINS}m"
done

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "Completed: $((TOTAL - FAILED))/${TOTAL} chromosomes"
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: ${FAILED} chromosome(s) failed. Check pipeline.log."
fi
ELAPSED=$(( $(date +%s) - START_TIME ))
echo "Total time: $(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m"
echo "============================================================"

echo ""
echo "Output files:"
ls -lhS ./hmm_input/chr*_hmm_data.npz 2>/dev/null || echo "  (none found)"
echo ""
echo "Breakpoint files:"
ls -lhS ./haptools_sim/sim_admixed_chr*.bp 2>/dev/null || echo "  (none found)"
