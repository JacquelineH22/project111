#!/bin/bash

set -e

# ================== User-set parameters  ==================

# Data directory and device type
prefix_general="Data/Xiamen-University/hnq/yb/demo_data/"  # 修改为你的demo_data路径
daughterCapture_folders=("D1" "D2" "D3" "D4" "D5")   # 修改为你实际的daughter capture文件夹名称
device="cuda"  # "cuda" or "cpu"

# Preprocessing parameters
pixel_size_raw=0.252  # current pixel size of raw large H&E mother image
pixel_size=0.5  # desired pixel size of large H&E mother image

# User selection 
n_genes=100  # number of most variable genes to impute (e.g. 1000 for Visium)
n_clusters=15 # number of clusters
dist_ST=100 # smoothing parameter across daughter ST samples

# ============================================================

export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

prefix="${prefix_general}MotherImage/"   

echo "=================================================="
echo "Starting iSCALE pipeline for demo_data"
echo "Data path: ${prefix_general}"
echo "Device: ${device}"
echo "=================================================="

############# Preprocess histology image #############
echo ""
echo "Step 1: Preprocessing histology image..."

# If your image needs rescaling, uncomment this:
# python rescale_img.py \
#     --prefix=${prefix} \
#     --pixelSizeRaw=${pixel_size_raw} \
#     --pixelSize=${pixel_size} \
#     --image \
#     --outputDir=${prefix}

python preprocess.py \
    --prefix=${prefix} \
    --image \
    --outputDir=${prefix}

echo "✓ Image preprocessing complete"

 
############# Daughter capture alignment to mother image #############
echo ""
echo "Step 2: Combining daughter capture data..."

# Combine data from n daughter captures (locs and cnts)
args=()
for d in "${daughterCapture_folders[@]}"; do
    args+=("${prefix_general}DaughterCaptures/AllignedToMother/${d}/")
done

python stitch_locs_cnts_relativeToM.py \
    "${prefix}" \
    "${args[@]}"

echo "✓ Daughter captures combined"


############# Visualize spot-level ST aligned to mother image #############
echo ""
echo "Step 3: Selecting genes and visualizing spots..."

# select most highly variable genes to predict
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

# visualize spot-level gene expression data
python plot_spots.py ${prefix} grayHE_flag=True
python plot_spots_integrated.py ${prefix} grayHE_flag=True ${dist_ST}

echo "✓ Gene selection and spot visualization complete"


############# Extract histology features from mother image #############
echo ""
echo "Step 4: Extracting histology features (this may take a while)..."

# extract histology features
python extract_features.py ${prefix} --device=${device}

# auto detect tissue mask
python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}
python refine_mask.py --prefix=${prefix} 
python plot_embeddings.py ${prefix}embeddings-hist.pickle ${prefix} --mask=${prefix}mask-small.png  

echo "✓ Feature extraction complete"


############# Predict super-resolution gene expression across mother image #############
echo ""
echo "Step 5: Training model and predicting gene expression (this will take a while)..."

# train gene expression prediction model and predict at super-resolution
python impute_integrated.py ${prefix} --epochs=1000 --device=${device}  --n-states=5  --dist=${dist_ST}
python refine_gene.py ${prefix} "conserve_index.pickle"  

echo "✓ Gene expression prediction complete"

# visualize imputed gene expression
echo ""
echo "Step 6: Visualizing imputed gene expression..."
python plot_imputed_iSCALE.py ${prefix}

# merge imputed gene expression 
python merge_imputed.py ${prefix} 1

echo "✓ Visualization complete"


############# Perform clustering based on super-resolution gene expression #############
echo ""
echo "Step 7: Performing clustering..."

# segment image by gene features
python cluster_iSCALE.py \
    --n-clusters=${n_clusters} \
    --filter-size=2 \
    --min-cluster-size=20 \
    --mask=${prefix}filterRGB/mask-small-refined.png \
    --refinedImage=${prefix}filterRGB/conserve_index.pickle \
    ${prefix}embeddings-gene.pickle \
    ${prefix}iSCALE_output/clusters-gene_${n_clusters}/

echo "✓ Clustering complete"

################### Model training information ###################
echo ""
echo "Step 8: Evaluating performance..."

# Evaluate performance (training)
python evaluate_fit.py ${prefix}

echo "✓ Evaluation complete"

################### Cell type annotation (optional) ###################
echo ""
echo "Step 9: Cell type annotation (if markers file exists)..."

# Annotation using marker list (if exists)
if [ -f "${prefix}markers_exampleFile.csv" ]; then
    python pixannot_percentile.py ${prefix} ${prefix}markers_exampleFile.csv ${prefix}/iSCALE_output/annotations/
    echo "✓ Annotation complete"
else
    echo "⚠ No markers file found, skipping annotation"
fi

echo ""
echo "=================================================="
echo "✓ iSCALE pipeline completed successfully!"
echo "Results saved to: ${prefix}iSCALE_output/"
echo "=================================================="

