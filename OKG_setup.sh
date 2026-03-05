#!/usr/bin/env bash
###############################################################################
# OKG_setup.sh — Set up the environment and directory structure for the
# LAI (Local Ancestry Inference) data pipeline.
#
# Prerequisites (system packages):
#   sudo apt install bcftools tabix git wget
#
# Usage:
#   bash OKG_setup.sh
###############################################################################

set -euo pipefail

echo "============================================================"
echo "LAI Data Pipeline — Environment Setup"
echo "============================================================"
echo ""

# ── 1. Create Python virtual environment ──────────────────────
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "[1/5] Virtual environment already exists at ${VENV_DIR}, skipping."
else
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
echo "      Installing Python packages..."
pip install --quiet haptools scikit-allel pandas numpy
echo "      Installed: haptools, scikit-allel, pandas, numpy"

# ── 2. Create directory structure ─────────────────────────────
echo "[2/5] Creating directory structure..."
mkdir -p 1kg_phase3_vcfs
mkdir -p haptools_input/reference_panels
mkdir -p haptools_input/reference_panels_thinned
mkdir -p haptools_input/admixed
mkdir -p haptools_input/sample_maps
mkdir -p haptools_input/logs
mkdir -p haptools_sim/genetic_maps
mkdir -p hmm_input

echo "      Directories created:"
echo "        1kg_phase3_vcfs/              (raw 1000 Genomes VCFs)"
echo "        haptools_input/               (filtered reference panels & admixed VCFs)"
echo "        haptools_sim/                 (simulation config, maps, outputs)"
echo "        hmm_input/                    (numpy arrays for HMM)"

# ── 3. Download GRCh37 genetic maps ──────────────────────────
if ls haptools_sim/genetic_maps/chr22.map &>/dev/null; then
    echo "[3/5] Genetic maps already exist, skipping."
else
    echo "[3/5] Downloading GRCh37 genetic maps..."
    cd haptools_sim/genetic_maps
    git clone --quiet https://github.com/adimitromanolakis/geneticMap-GRCh37.git tmp_maps

    for chr in {1..22}; do
        # Decompress
        gunzip -c tmp_maps/genetic_map_GRCh37_chr${chr}.txt.gz > chr${chr}.raw
        # Remove header, strip 'chr' prefix, reorder to PLINK format: chr . cM bp
        tail -n +2 chr${chr}.raw \
            | sed 's/^chr//' \
            | awk '{print $1, ".", $4, $2}' OFS='\t' > chr${chr}.map
        rm chr${chr}.raw
    done

    rm -rf tmp_maps
    cd ../..
    echo "      22 genetic map files created in haptools_sim/genetic_maps/"
fi

# ── 4. Create admixture model file ───────────────────────────
MODEL_FILE="haptools_sim/afr_eur_admixture.dat"
if [ -f "$MODEL_FILE" ]; then
    echo "[4/5] Admixture model already exists, skipping."
else
    echo "[4/5] Creating admixture model (80% YRI / 20% CEU, 6 generations)..."
    printf '100\tAdmixed\tYRI\tCEU\n' > "$MODEL_FILE"
    printf '1\t0\t0.8\t0.2\n' >> "$MODEL_FILE"
    for gen in 2 3 4 5 6; do
        printf '%d\t1\t0\t0\n' "$gen" >> "$MODEL_FILE"
    done
    echo "      Created: ${MODEL_FILE}"
fi

# ── 5. Verify tools ──────────────────────────────────────────
echo "[5/5] Verifying tools..."
MISSING=0
for cmd in bcftools tabix haptools python3; do
    if command -v "$cmd" &>/dev/null; then
        VERSION=$("$cmd" --version 2>&1 | head -1)
        echo "      $cmd: $VERSION"
    else
        echo "      WARNING: $cmd not found!"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "============================================================"
echo "Setup complete."
echo ""
echo "Next steps (run in order):"
echo "  1. bash OKGDL.sh                      # Download 1000G VCFs (~15GB, ~2 hours)"
echo "  2. bash OKG_vcfs_to_haptools.sh        # Filter into reference panels (~3 hours)"
echo "  3. nohup bash OKG_run_all_chroms.sh \\  # Thin, simulate, convert (~4-8 hours)"
echo "       > pipeline.log 2>&1 &"
echo "============================================================"
