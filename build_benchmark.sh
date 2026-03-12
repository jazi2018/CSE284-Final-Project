#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="./data"
OUT_DIR="benchmark_chr21"

echo "===================================="
echo "Building benchmark dataset"
echo "===================================="

mkdir -p "${OUT_DIR}"

python build_benchmark_inputs.py \
  --npz "${DATA_DIR}/chr21_hmm_data.npz" \
  --ref-samples "${DATA_DIR}/ref_samples.txt" \
  --query-samples "${DATA_DIR}/query_samples.txt" \
  --igsr "${DATA_DIR}/igsr_samples.tsv" \
  --bp "${DATA_DIR}/sim_admixed_chr21.bp" \
  --outdir "${OUT_DIR}" \
  --chrom 21

echo ""
echo "===================================="
echo "Benchmark files created in:"
echo "${OUT_DIR}"
echo "===================================="

ls -lh "${OUT_DIR}"