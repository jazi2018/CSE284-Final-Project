#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data"
BENCH_DIR="${PROJECT_DIR}/benchmark_chr21_small"
FULL_BENCH_DIR="${PROJECT_DIR}/benchmark_chr21"
RESULTS_DIR="${PROJECT_DIR}/benchmark_results_chr21"
SCORED_DIR="${PROJECT_DIR}/benchmark_scored_chr21"

FLARE_JAR="${PROJECT_DIR}/flare.jar"

PYTHON_BIN="python"
R_BIN="Rscript"

mkdir -p "${BENCH_DIR}" "${RESULTS_DIR}" "${SCORED_DIR}"

echo "===================================="
echo "Step 1: Ensure benchmark inputs exist"
echo "===================================="

mkdir -p "${FULL_BENCH_DIR}" "${RESULTS_DIR}" "${SCORED_DIR}"

REF_PANEL_FILE="${BENCH_DIR}/reference_panel.npy"
STUDY_VCF_FILE="${BENCH_DIR}/study.vcf"

if [[ -f "${REF_PANEL_FILE}" && -f "${STUDY_VCF_FILE}" ]]; then
    echo "Benchmark inputs already exist in ${BENCH_DIR} — skipping Step 1."
else
    echo "Benchmark inputs missing in ${BENCH_DIR}."

    # If BENCH_DIR is the full benchmark directory, build from raw inputs
    if [[ "${BENCH_DIR}" == "${FULL_BENCH_DIR}" ]]; then
        echo "Building full benchmark inputs from raw data..."
        mkdir -p "${FULL_BENCH_DIR}"

        "${PYTHON_BIN}" build_benchmark_inputs.py \
          --npz "${DATA_DIR}/chr21_hmm_data.npz" \
          --ref-samples "${DATA_DIR}/ref_samples.txt" \
          --query-samples "${DATA_DIR}/query_samples.txt" \
          --igsr "${DATA_DIR}/igsr_samples.tsv" \
          --bp "${DATA_DIR}/sim_admixed_chr21.bp" \
          --outdir "${FULL_BENCH_DIR}" \
          --chrom 21
    else
        echo "Subset benchmark inputs requested."
        echo "Building full benchmark first if needed..."

        if [[ ! -f "${FULL_BENCH_DIR}/reference_panel.npy" || ! -f "${FULL_BENCH_DIR}/study.vcf" ]]; then
            mkdir -p "${FULL_BENCH_DIR}"

            "${PYTHON_BIN}" build_benchmark_inputs.py \
              --npz "${DATA_DIR}/chr21_hmm_data.npz" \
              --ref-samples "${DATA_DIR}/ref_samples.txt" \
              --query-samples "${DATA_DIR}/query_samples.txt" \
              --igsr "${DATA_DIR}/igsr_samples.tsv" \
              --bp "${DATA_DIR}/sim_admixed_chr21.bp" \
              --outdir "${FULL_BENCH_DIR}" \
              --chrom 21
        fi

        echo "Creating subset benchmark inputs..."
        "${PYTHON_BIN}" subset_benchmark_inputs.py \
          --input-dir "${FULL_BENCH_DIR}" \
          --output-dir "${BENCH_DIR}" \
          --max-ref-samples 50 \
          --max-query-samples 10 \
          --max-snps 20000
    fi
fi


echo
echo "===================================="
echo "Step 2: Run your method"
echo "===================================="

"${PYTHON_BIN}" run_my_method.py \
  --reference-panel-npy "${BENCH_DIR}/reference_panel.npy" \
  --labels-tsv "${BENCH_DIR}/reference_labels.tsv" \
  --ancestry-index-tsv "${BENCH_DIR}/ancestry_index.tsv" \
  --study-vcf "${BENCH_DIR}/study.vcf" \
  --out-tsv-gz "${RESULTS_DIR}/my_method.tsv.gz" \
  --recomb-prob 0.01 \
  --admixture-prob 0.001 \
  --error-rate 0.01


echo
echo "===================================="
echo "Step 3: Run FLARE"
echo "===================================="

if [ ! -f "${FLARE_JAR}" ]; then
  echo "ERROR: Could not find FLARE jar at ${FLARE_JAR}"
  exit 1
fi

"${PYTHON_BIN}" run_flare.py \
  --flare-jar "${FLARE_JAR}" \
  --ref-vcf "${BENCH_DIR}/reference.vcf" \
  --ref-panel "${BENCH_DIR}/ref_panel.tsv" \
  --study-vcf "${BENCH_DIR}/study.vcf" \
  --map "${BENCH_DIR}/genetic_map.tsv" \
  --out-prefix "${RESULTS_DIR}/flare_run" \
  --parsed-out-tsv-gz "${RESULTS_DIR}/flare.tsv.gz" \
  --ancestry-index-tsv "${BENCH_DIR}/ancestry_index.tsv" \
  --java-mem-gb 8 \
  --nthreads 4 \
  --seed 1 \
  --probs \
  --extra-args min-mac=1 min-maf=0

echo
echo "===================================="
echo "Step 4: Score methods"
echo "===================================="

"${PYTHON_BIN}" score_methods.py \
  --truth-tsv-gz "${BENCH_DIR}/truth.tsv.gz" \
  --prediction-files "${RESULTS_DIR}/my_method.tsv.gz" "${RESULTS_DIR}/flare.tsv.gz" \
  --outdir "${SCORED_DIR}"


echo
echo "===================================="
echo "Step 5: Make figures"
echo "===================================="

"${R_BIN}" plot_benchmark.R "${SCORED_DIR}"


echo
echo "===================================="
echo "Done"
echo "===================================="

echo "Inputs:   ${BENCH_DIR}"
echo "Results:  ${RESULTS_DIR}"
echo "Scored:   ${SCORED_DIR}"
echo "Figures:  ${SCORED_DIR}/figures"

echo
echo "Overall summary:"
cat "${SCORED_DIR}/overall_summary.tsv"