# run_iscale (iSCALE on Xenium pseudo-Visium)

This folder contains helper scripts to run **iSCALE** on **Xenium pseudo-Visium** data in the format expected by the upstream iSCALE code under `./iSCALE/`.

## Key entry points

- `convert_to_iscale_input.py`  
  Convert **one** Xenium pseudo-Visium `*.h5ad` + one H&E image into an iSCALE input project with a **single** daughter capture (`D1`).

- `split_xenium_into_captures.py`  
  Split a **full-slide** Xenium pseudo-Visium dataset into **multiple** daughter captures (`D1..Dn`) using a paper-like grid (default: 3.2mm × 3.2mm tiles), then save in iSCALE input format.

- `xenium_benchmark_pipeline.py`  
  One-shot “benchmark-style” pipeline: split captures → write training input → save full-slide ground-truth → generate a minimal evaluation script template + capture grid visualization.

- `run_iscale_xenium.sh`  
  The main runner script that executes the iSCALE pipeline steps (preprocess → stitch captures → features → mask → train/impute → clustering → evaluate_fit).

## What iSCALE expects as input

`run_iscale_xenium.sh` (when executed from `./iSCALE/`) assumes:

- **Input**: `./data/iscale_input`
- **Output**: `./results/iscale`

Expected directory layout:

```text
iSCALE/
└── data/
    └── iscale_input/
        ├── DaughterCaptures/
        │   └── AllignedToMother/
        │       ├── D1/
        │       │   ├── cnts.tsv
        │       │   └── locs.tsv
        │       └── D2..Dn/ (optional)
        └── MotherImage/
            ├── he-raw.png
            └── radius-raw.txt
```

Notes:
- `locs.tsv` must contain spot coordinates in pixels (`x`, `y`).
- The `*.h5ad` used by the converters must contain `obs['x_pixel']` and `obs['y_pixel']`.

## Minimal run (recommended)

### 0) Environment and checkpoints

- Create the environment using `environment.yml` or `requirements.txt` in this folder.
- Download iSCALE checkpoints (HIPT/ViT `*.pth`) into `./iSCALE/checkpoints/`.

### 1) Generate iSCALE input (choose one)

#### A. Single capture (D1)

Edit the path variables near the top of `convert_to_iscale_input.py`:
- `H5AD_PATH`
- `HE_PATH`
- `OUTPUT_BASE` (recommended target: `.../run_iscale/iSCALE/data/iscale_input`)

Run:

```bash
python convert_to_iscale_input.py
```

#### B. Multi-capture (D1..Dn)

Edit the parameters at the bottom of `split_xenium_into_captures.py` (paths + tile size), then run:

```bash
python split_xenium_into_captures.py
```

### 2) Run iSCALE

`run_iscale_xenium.sh` calls iSCALE scripts like `preprocess.py` by **relative name**, so run it from the `./iSCALE/` directory:

```bash
cd /path/to/benchmarks/algorithms/run_iscale/iSCALE
chmod +x ../run_iscale_xenium.sh
CUDA_VISIBLE_DEVICES=0 nohup bash ../run_iscale_xenium.sh > iscale_run.log 2>&1 &
tail -f iscale_run.log
```

Common knobs are at the top of `run_iscale_xenium.sh`:
- `INPUT_BASE`, `OUTPUT_BASE`
- `N_GENES`, `EPOCHS`, `DIST_ST`, `N_CLUSTERS`, `DEVICE`

## When to use `xenium_benchmark_pipeline.py`

Use it if you want the “paper benchmark” style outputs (training input + ground truth + an evaluation template):

```bash
python xenium_benchmark_pipeline.py \
  --h5ad /path/to/simulated_data.h5ad \
  --he_image /path/to/he.png \
  --output /path/to/xenium_benchmark_out
```

It will create:
- `iscale_training_data/` (can be used as `INPUT_BASE` for `run_iscale_xenium.sh`)
- `ground_truth/`
- `evaluation/` (template evaluation script + `captures_grid.png`)

