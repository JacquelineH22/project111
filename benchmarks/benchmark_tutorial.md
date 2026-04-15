# Benchmark Tutorial

## Compared Methods

SpatioCell is compared with a wide range of ST annotation methods, including:

1. Deconvolution-based methods (e.g., Spotiphy, SpatialScope, Tangram, and CytoSPACE)
2. Histology-transcriptomics integration frameworks (e.g., iStar, iSCALE, TESLA, GHIST, and PanoSpace)
3. GigaTIME

For fair comparison, please use the official implementations of each method:

1. SpatialScope: https://github.com/YangLabHKUST/SpatialScope
2. Tangram: https://github.com/broadinstitute/Tangram
3. CytoSPACE: https://github.com/digitalcytometry/cytospace
4. iStar: https://github.com/daviddaiweizhang/istar
5. iSCALE: https://github.com/amesch441/iSCALE
6. GHIST: https://github.com/SydneyBioX/GHIST
7. TESLA: https://github.com/jianhuupenn/TESLA
8. PanoSpace: https://github.com/hehuifeng/PanoSpace
9. GigaTIME: https://github.com/prov-gigatime/GigaTIME

### Notes on Method Adaptation

- **Spotiphy**: The original implementation only supports ST inference with circular spots. We modified it to support square spots.
- **GHIST and related outputs**:
  - For methods that mainly generate single-cell expression (e.g., GHIST), we use TACCO for cell-type annotation.
  - For methods that generate super-resolution expression, we first aggregate bins into cells and then annotate cells with TACCO.
- **iSCALE**: The original method is designed for consecutive slices. We made modifications for our setting; see details in `run_iscale/RUN_ISCALE_DOC.md`.

```bash
cd utils
python bin2cell.py
python run_tacco.py
```

## Evaluation

Evaluation focuses on both cell segmentation quality and cell-type annotation accuracy.

Segmented cell (nucleus) centroids are mapped to ground-truth cell centroids using Mutual Nearest Neighbors (MNN).

First, prepare prediction and ground-truth `.csv` files. Each file should contain the columns `x`, `y`, and `label`, corresponding to x-coordinate, y-coordinate, and cell-type annotation, respectively.

Run:

```bash
cd metrics
python ct_annotate_eval.py --pred_file PRED_LABEL_PATH --gt_file GROUNDTRUTH_LABEL_PATH --savepath PATH_TO_SAVE_METRICS --logfile
```

