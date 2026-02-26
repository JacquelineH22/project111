# SpatioCell: Deep Integration of Histology and Spatial Transcriptomics for Profiling the Cellular Microenvironment at Single-Cell Level

SpatioCell is a computational algorithm to automatically extract both cell type and expression information at single-cell resolution from ST data, through a morpho-transcriptomic spatial reconstruction framework solved via dynamic programming, integrating morphological and transcriptomic information.

SpatioCell comprises two key modules: (1) a specialized spatio-morphological learner for histopathological image processing, which performs nuclei segmentation and detailed morphological feature extraction, and (2) a morpho-transcriptomic spatial reconstruction module for cell annotation that integrates imagederived morphological features with transcriptomic data to achieve high-precision, single-cell–level annotation

## Set Up Environment

```shell
conda create -n SpatioCell python=3.8 -y
conda activate SpatioCell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements.txt
```

## Module 1: Nuclei Segmentation and Classification

We developed a novel nuclei segmentation and analysis model that combines a multi-scale feature encoding automatic prompt generator (Prompter) with a fine-tuned SAM model adapted for H&E images. This approach leverages H&E images for precise cell localization and counting, while incorporating cell morphology to generate classification probability profiles.

### Train

#### Extract Patches and Preprocess

Extract patches and format data using `extract_patches.py` and `preprocess.py`

```shell
cd data_prepare
python extract_patches.py
python preprocess.py
```

#### Train SAM

Modified the config files in `configs`:

* Set path to the train, valid and test datasets.
* Set number of nuclei types in the train dataset.
* Set path where checkpoints will be saved.
* Set path to the pretrained checkpoint.

Run the following command to fintune SAM model.

```shell
cd morphology_analyzer
CUDA_VISIBLE_DEVICES=0 python train.py configs/type/consep-train-sam.py
```

#### Train Prompter

Run the following command to train the prompter.

```shell
CUDA_VISIBLE_DEVICES=0 python train.py /configs/type/consep-train-prompt.py
```

### Infer

Run inference using the following command. Results will be saved in `outs/` directory.

```shell
CUDA_VISIBLE_DEVICES=0 python test.py \
    configs/type/pannuke-train-prompt.py \
    work_dirs/type/pannuke/train-prompt/latest.pth \
    --format-only --eval-options \
    imgfile_prefix=outs/
```

### Evaluation

Finally, calculate evaluation metrics, including DICE, AJI, DQ, SQ and PQ  through  `compute_stats.py`

```shell
cd metrics
python compute_stats.py --mode instance --pred_dir outs --true_dir /path/to/gt_masks/ --ext mat 
```

## Module 2: Cell Type Annoataion

To overcome the limitation of multicellular resolution, the cell annotation module of SpatioCell combines H&E-derived and expression-derived information, enabling the construction of high-resolution single-cell spatial maps and improving the accuracy of cell-type assignment in STs.

### Input Files

* nuclei_segmentation_results.json: Output of SpatioCell nuclei segmentation and classification module.
* nuclei_segmentation_types.json: Nuclei classification types of SpatioCell.
* type_mapping.json: A mapping from H&E-based nuclei classification types to deconvolution types, saved in json format.
* deconvolution_results.csv: Deconvolution results, a table containing cell abundance in each spot, with spot barcodes in rows and cell types in columns.
* tissue_positions.csv, scalefactors_json.json: Outputs of CellRanger, containing spot position and scale factors of images.
* image.tif: Full resolution image of H&E stained slide, used for nuclei segmentation and classification, and visualization of cell type annotation results.

### Parameters

Data path and other parameters should be set in `celltype_assign/configs.yaml`.

The annotation relies on deconvolution-based constraints for cell annotation, but the inherent noise of ST and single-cell reference data—caused by molecular diffusion, sequencing errors, and uneven sampling—make cell-type abundance estimates from deconvolution sometimes deviate from true composition in complex regions.

Here, we introduce Competitive Balance Index (CBI)—a parameter that quantifies the discrepancy between deconvolution results and H&E-derived nuclei classification, to correct errors that arise from deconvolution. Specifically, when CBI exceeds a pre-set threshold, SpatioCell progressively refines the deconvolution result using H&E-derived classification probabilities until the discrepancy falls below the threshold.

Set `cbi_threshold` in `celltype_assign/configs.yaml` to apply CBI to refine cell type assignment results.

### Annotate Cell Types

Run the following commands to annotate cell types using SpatioCell:

```shell
cd celltype_assign
python run_celltype_assign.py
```

You can visualize the SpatioCell results using `celltype_assign/tutorial.ipynb`.

In addition, SpatioCell uses [Spotiphy](https://github.com/jyyulab/Spotiphy) as the default deconvolution backend. We also provide several commonly used alternatives, including Cell2Location, CytoSpace, and SpatialScope, in the `deconvolution_backends` directory.

## Module 3: Benchmarks

1. Simulated dataset generation

Scripts for generating simulated multicellular-resolution spatial transcriptomics (ST) data from Xenium ST data are provided in the `data_prepare` directory.

First, download the Xenium dataset from:  
`https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast`

Then run:

```shell
cd data_prepare/
python simulate_visium.py
```

The output file `simulated_data.h5ad` contains simulated single-cell–level expression matrices, cell centroid positions derived from DAPI, and cell annotations (if provided).

2. Annotation with SpatioCell

Follow the instructions above to obtain single-cell spatial maps from the simulated BRCA ST data, and then perform cell-type annotation using SpatioCell.

3. Annotation with competing methods

Scripts for running competing methods, including Spotiphy, iStar, Tesla, GHIST, etc., are provided in `benchmarks/algorithms`.

4. Metrics

Scripts for evaluation metrics are provided in `benchmarks/metrics/ct_annotate_eval.py`.
For SpatioCell's annotation output in JSON format, first convert the results to a `.csv` file with the following columns: `cell_id`, `x`, `y`, and `cell_type`.  
Then run:

```shell
cd benchmarks/metrics
python ct_annotate_eval.py \
  --pred_file /path/to/pred_csv_file \
  --gt_file /path/to/gt_csv_file \
  --savepath output/metrics.pkl \
  --logfile output/metrics.log
```



