#!/bin/bash
#BSUB -J iSCALE_run_demo                   # Job name
#BSUB -q gpu_xxx                           # Queue name
#BSUB -n 1                                 # CPU cores (increase if using multi with >1 workers)
#BSUB -gpu "num=1"                         # Request 1 GPU
#BSUB -R "rusage[mem=64000]"               # Memory (MB)
#BSUB -W 100:00                            # Walltime
#BSUB -o logs/logs_output/hs_output_%J.log # Stdout
#BSUB -e logs/logs_errors/hs_error_%J.log  # Stderr


set -e

# ================== User-set parameters  ==================

# Data directory and device type
prefix_general="Data/demo/"  # e.g. Data/demo/ **** Note: must have subfolders "DaughterCaptures" and "MotherImage" ****
daughterCapture_folders=("D1" "D2" "D3" "D4" "D5")   # list of subfolders in DaughterCaptures
device="cuda"  # "cuda" or "cpu"

# Preprocessing parameters
pixel_size_raw=0.252  # current pixel size of raw large H&E mother image
pixel_size=0.5  # desired pixel size of large H&E mother image

# User selection 
n_genes=100  # number of most variable genes to impute (e.g. 1000)
n_clusters=15 # number of clusters
dist_ST=100 # smoothing parameter across daughter ST samples

# ============================================================

export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

prefix="${prefix_general}MotherImage/"  

############# Create iSCALE environment #############

#conda env create -f environment.yml
#conda activate iSCALE_env

############# Preprocess histology image #############

#### Scale and pad image
# If your image has already been scaled to desired resolution, you can keep the
# rescale_img.py script commented out and label your image he-scaled.*

#python rescale_img.py \
#    --prefix=${prefix} \
#    --pixelSizeRaw=${pixel_size_raw} \
#    --pixelSize=${pixel_size} \
#    --image \
#    --outputDir=${prefix}

python preprocess.py \
    --prefix=${prefix} \
    --image \
    --outputDir=${prefix}

 
############# Daughter capture alignment to mother image #############

#### Step 1: 
# First, please use Alignment_scripts/AlignmentMethod.ipynb jupyter notebook to 
# perform semi-automatic allignment of daughter captures to mother image. 
# Place data in "AllignedToMother" folder

# === Step 2: Combine data from n daughter captures (locs and cnts) ===
args=()
for d in "${daughterCapture_folders[@]}"; do
    args+=("${prefix_general}DaughterCaptures/AllignedToMother/${d}/")
done

python stitch_locs_cnts_relativeToM.py \
    "${prefix}" \
    "${args[@]}"

### if you are running each python script individually in the terminal, you may use the following command to 
### combine data from n daughter captures: (changing D1, D2, D3, .... to your daughter capture folder names)
# python stitch_locs_cnts_relativeToM.py ${prefix} ${prefix_general}DaughterCaptures/AllignedToMother/D1/ ${prefix_general}DaughterCaptures/AllignedToMother/D2/ ${prefix_general}DaughterCaptures/AllignedToMother/D3/ ${prefix_general}DaughterCaptures/AllignedToMother/D4/ ${prefix_general}DaughterCaptures/AllignedToMother/D5/ 


############# Visualize spot-level ST aligned to mother image #############

#### select most highly variable genes to predict
#### If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

#### rescale coordinates and spot radius (if aligned to full-res and not scaled image)
#python rescale_locs.py.py ${prefix} --locs --radius

#### visualize spot-level gene expression data
python plot_spots.py ${prefix} grayHE_flag=True
python plot_spots_integrated.py ${prefix} grayHE_flag=True ${dist_ST}


############# Extract histology features from mother image #############

#### extract histology features
python extract_features.py ${prefix} --device=${device}
#### # If you want to retun model, you need to delete the existing results:
# rm ${prefix}embeddings-hist-raw.pickle

#### auto detect tissue mask
python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}
python refine_mask.py --prefix=${prefix} 
python plot_embeddings.py ${prefix}embeddings-hist.pickle ${prefix} --mask=${prefix}mask-small.png  

#### segment image by histology features
#python cluster_HE.py --n-clusters=n_clusters ${prefix}embeddings-histOnly.pickle ${prefix}


############# Predict super-resolution gene expression across mother image #############

#### train gene expression prediction model and predict at super-resolution
python impute_integrated.py ${prefix} --epochs=1000 --device=${device}  --n-states=5  --dist=${dist_ST} # train model from scratch
python refine_gene.py ${prefix} "conserve_index.pickle"  

##### # If you want to retrain model, you need to delete the existing model:
# rm -r ${prefix}states

#### visualize imputed gene expression
python plot_imputed_iSCALE.py ${prefix}

#### merge imputed gene expression 
python merge_imputed.py ${prefix} 1 #can change 1 for varying resolution


############# Perform clustering based on super-resolution gene expression #############

#### segment image by gene features
python cluster_iSCALE.py \
    --n-clusters=${n_clusters} \
    --filter-size=2 \
    --min-cluster-size=20 \
    --mask=${prefix}filterRGB/mask-small-refined.png \
    --refinedImage=${prefix}filterRGB/conserve_index.pickle \
    ${prefix}embeddings-gene.pickle \
    ${prefix}iSCALE_output/clusters-gene_${n_clusters}/

################### Model training information ###################

#### Evaluate performance (training)
python evaluate_fit.py ${prefix} ## training rmse and pearson


################### Cell type annotation for whole tissue ###################

### Annotation using example marker list markers_example.csv
python pixannot_percentile.py ${prefix} ${prefix}markers_exampleFile.csv ${prefix}/iSCALE_output/annotations/




