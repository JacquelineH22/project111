#!/bin/bash
################################################################################
# iSCALE Pipeline - Xenium Pseudo-Visium Data
#
# Usage:
#   chmod +x run_iscale_xenium.sh
#   CUDA_VISIBLE_DEVICES=2 nohup bash run_iscale_xenium.sh > iscale_run.log 2>&1 &
#   tail -f iscale_run.log
################################################################################

set -e

# ==================== Configuration ====================

INPUT_BASE="./data/iscale_input"
OUTPUT_BASE="./results/iscale"

DEVICE="cuda"
N_GENES=313
N_CLUSTERS=15
DIST_ST=100
N_STATES=5
EPOCHS=1000
PIXEL_SIZE_RAW=0.2125
PIXEL_SIZE=0.5

export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32

# ==================== Helpers ====================

step() { echo; echo "[$1] $2"; echo "----------------------------------------"; }

check_file() { [ -f "$1" ] || { echo "ERROR: missing file $1"; exit 1; }; }
check_dir()  { [ -d "$1" ] || { echo "ERROR: missing dir $1";  exit 1; }; }

# ==================== Validate Inputs ====================

echo "Python: $(python --version 2>&1)"
[[ "$CONDA_DEFAULT_ENV" == "iSCALE_env" ]] || echo "WARNING: not in iSCALE_env (current: $CONDA_DEFAULT_ENV)"

check_dir  "$INPUT_BASE"
check_dir  "$INPUT_BASE/MotherImage"
check_dir  "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1"
check_file "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1/cnts.tsv"
check_file "$INPUT_BASE/DaughterCaptures/AllignedToMother/D1/locs.tsv"

HE_FILE=$(find "$INPUT_BASE/MotherImage" -name "he-raw.*" -o -name "he-scaled.*" | head -1)
[ -n "$HE_FILE" ] || { echo "ERROR: H&E image not found"; exit 1; }

mkdir -p "$OUTPUT_BASE/MotherImage"

# Convenience aliases
prefix_mother="${OUTPUT_BASE}/MotherImage/"
input_mother="${INPUT_BASE}/MotherImage/"

echo "Input:  $INPUT_BASE"
echo "Output: $OUTPUT_BASE"

START_TIME=$(date +%s)

# ==================== Step 1: Image Preprocessing ====================

step "1/10" "Image preprocessing"

cp "$INPUT_BASE/MotherImage/he-raw."* "${OUTPUT_BASE}/MotherImage/" 2>/dev/null || true

[ -f "$INPUT_BASE/MotherImage/radius-raw.txt" ] && \
    cp "$INPUT_BASE/MotherImage/radius-raw.txt" "${OUTPUT_BASE}/MotherImage/"

python preprocess.py \
    --prefix="${prefix_mother}" \
    --image \
    --outputDir="${prefix_mother}"

# Compute scaled radius (must be integer)
if [ -f "${OUTPUT_BASE}/MotherImage/radius-raw.txt" ]; then
    RADIUS_RAW=$(cat "${OUTPUT_BASE}/MotherImage/radius-raw.txt")
    RADIUS_SCALED=$(python -c "print(int(${RADIUS_RAW} * ${PIXEL_SIZE_RAW} / ${PIXEL_SIZE} + 0.5))")
else
    RADIUS_SCALED=$(python -c "print(int((55/2) / ${PIXEL_SIZE} + 0.5))")
fi
echo "${RADIUS_SCALED}" > "${OUTPUT_BASE}/MotherImage/radius.txt"
echo "radius.txt: ${RADIUS_SCALED} px"

# ==================== Step 2: Merge All Daughter Captures ====================

step "2/10" "Merging Daughter Captures"

CAPTURE_PATHS=()
for cap_dir in "${INPUT_BASE}/DaughterCaptures/AllignedToMother/D"*/; do
    [ -d "$cap_dir" ] && CAPTURE_PATHS+=("$cap_dir")
done
echo "Found ${#CAPTURE_PATHS[@]} captures"

python stitch_locs_cnts_relativeToM.py \
    "${prefix_mother}" \
    "${CAPTURE_PATHS[@]}"

echo "Total spots: $(($(wc -l < ${prefix_mother}locs.tsv) - 1))"

# ==================== Step 3: Gene Selection ====================

step "3/10" "Selecting highly variable genes (n=${N_GENES})"

python select_genes.py \
    --n-top=${N_GENES} \
    "${prefix_mother}cnts.tsv" \
    "${prefix_mother}gene-names.txt"

# ==================== Step 4: Spot Visualization ====================

step "4/10" "Visualizing spots"

python plot_spots.py \
    "${prefix_mother}" \
    grayHE_flag=True

python plot_spots_integrated.py \
    "${prefix_mother}" \
    grayHE_flag=True \
    ${DIST_ST}

# ==================== Step 5: Histology Feature Extraction ====================

step "5/10" "Extracting histology features (HIPT)"

python extract_features.py \
    "${prefix_mother}" \
    --device=${DEVICE}

# ==================== Step 6: Tissue Mask ====================

step "6/10" "Generating tissue mask"

python get_mask.py \
    "${prefix_mother}embeddings-hist.pickle" \
    "${prefix_mother}"

python refine_mask.py \
    --prefix="${prefix_mother}"

python plot_embeddings.py \
    "${prefix_mother}embeddings-hist.pickle" \
    "${prefix_mother}" \
    --mask="${prefix_mother}mask-small.png"

# ==================== Step 7: Train & Predict ====================

step "7/10" "Training gene expression model (epochs=${EPOCHS}, states=${N_STATES})"

python impute_integrated.py \
    "${prefix_mother}" \
    --epochs=${EPOCHS} \
    --device=${DEVICE} \
    --n-states=${N_STATES} \
    --dist=${DIST_ST}

python refine_gene.py \
    "${prefix_mother}" \
    "conserve_index.pickle"

# ==================== Step 8: Result Visualization ====================

step "8/10" "Visualizing predictions"

python plot_imputed_iSCALE.py \
    "${prefix_mother}"

python merge_imputed.py \
    "${prefix_mother}" \
    1

# ==================== Step 9: Clustering ====================

step "9/10" "Clustering (k=${N_CLUSTERS})"

python cluster_iSCALE.py \
    --n-clusters=${N_CLUSTERS} \
    --filter-size=2 \
    --min-cluster-size=20 \
    --mask="${prefix_mother}filterRGB/mask-small-refined.png" \
    --refinedImage="${prefix_mother}filterRGB/conserve_index.pickle" \
    "${prefix_mother}embeddings-gene.pickle" \
    "${prefix_mother}iSCALE_output/clusters-gene_${N_CLUSTERS}/"

# ==================== Step 10: Evaluation ====================

step "10/10" "Evaluating model performance"

python evaluate_fit.py \
    "${prefix_mother}"

# ==================== Done ====================

ELAPSED=$(( $(date +%s) - START_TIME ))
printf "\nDone in %dh %dm %ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Output: ${OUTPUT_BASE}"