<p align="left">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_fig1_top.png" width="300"/>
</p>


# Scaling up spatial transcriptomics for large-sized tissues with **iSCALE**


<p align="center">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_fig1_.png" width="1200"/>
</p>
<p align="center">
  <strong>Figure:</strong> <em>iSCALE workflow</em>
</p>

<p align="justify">
<b>iSCALE</b> (<i>Inferring Spatially resolved Cellular Architectures for Large-sized tissue Environments</i>)  
is a novel framework designed to integrate multiple daughter captures and utilize H&amp;E information from large tissue samples, enabling prediction of gene expression with near single-cell resolution across whole-slide tissues.
</p>


---

## 🔧 Installation & Setup

Clone the repository (recommended) or download the `.zip` directly from GitHub:

```bash
git clone https://github.com/amesch441/iSCALE.git
cd iSCALE-main
```

### Option A (recommended): Conda
```bash
conda env create -f environment.yml
conda activate iSCALE_env
cd iSCALE
```

### Option B: Pip
```bash
python -m venv iSCALE_env
source iSCALE_env/bin/activate   # Linux/Mac
# or: .\iSCALE_env\Scripts\activate   # Windows
pip install -r requirements.txt
cd iSCALE
```

> ⚡ **GPU usage is strongly recommended** for speed and scalability. CPU mode is supported but slower.

---

## 📦 Download Demo Data & Checkpoints

Download from [Box link](https://upenn.box.com/s/cburekr425ibu276wyxki09q35z2o3x0).

- Place the model checkpoint files:
  - `vit4k_xs_dino.pth`
  - `vit256_small_dino.pth`  
  into:
  ```
  iSCALE-main/iSCALE/checkpoints/
  ```

- Place the `demo` folder into:
  ```
  iSCALE-main/iSCALE/data/
  ```

---

## ▶️ Running iSCALE

To run the demo, submit the appropriate job script depending on your cluster scheduler:

```bash
bsub < _run_iSCALE_bsub.sh     # For LSF systems
sbatch _run_iSCALE_sbatch.sh   # For SLURM systems
```

with `prefix="Data/demo/"`.  
Ground truth for this demo gastric tumor tissue can be found in the `cnts-truth-agg` folder.

- Use `_run_iSCALE_sbatch.sh` if your system uses **SLURM**.  
- Use `_run_iSCALE_bsub.sh` if your system uses **LSF**.  
  (These scripts are identical except for scheduler setup.)

<p align="justify">
⚠️ <b>Important</b>: Make sure to edit the header of the run script (<code>#SBATCH</code> for SLURM or <code>#BSUB</code> for LSF) to set the correct queue/partition name for your system, as well as any resource requests (GPUs, memory, runtime).
</p>
---

## 📂 Repository Structure

```
iSCALE-main/
│
├── environment.yml         # conda environment specification
├── requirements.txt        # pip requirements
│
├── iSCALE/
│   ├── checkpoints/        # pretrained models (place downloaded .pth files here)
│   ├── data/               # input data (demo folder goes here)
│   ├── Alignment_scripts/  # tools for semi-automatic alignment
│   ├── logs/               # log directory
│   │   ├── logs_output/    # job standard output logs
│   │   └── logs_errors/    # job error logs
│   ├── *.py                # main Python scripts
│   ├── *.sh                # run scripts (SLURM/LSF)
│   └── ...
```


---

## 📂 Input Data & Formats

Each project has the following structure:

```
iSCALE-main/iSCALE/Data/<project_name>/
│
├── DaughterCaptures/
│   ├── UnallignedToMother/        # raw ST data (Visium, Visium HD, Xenium, CosMx)
│   │   ├── D1/
│   │   │   ├── cnts.tsv           # count matrix (genes × spots)
│   │   │   ├── locs.tsv           # coordinates (spot_id, x, y)
│   │   │   └── he.*               # H&E image (see formats below)
│   │   ├── D2/
│   │   └── ...
│   │
│   └── AllignedToMother/          # aligned data (produced after registration)
│       ├── D1/
│       │   ├── cnts.tsv
│       │   └── locs.tsv
│       ├── D2/
│       └── ...
│
└── MotherImage/
    ├── he-raw.*                   # raw H&E (before scaling)
    ├── he-scaled.*                # scaled H&E (after resizing)
    ├── he.tiff                    # final processed H&E with padding
    ├── radius-raw.txt             # raw spot radius in pixels
    ├── radius.txt                 # scaled radius in pixels (auto-generated if missing using rescale_locs.py)
    └── markers.csv (optional)     # marker genes for auto-annotation
```

### Notes
- **Always run `preprocess.py` to generate the final `he.tiff` file** for the MotherImage folder.  
- **Supported input H&E formats for mother image**:  
  `.tiff`, `.tif`, `.svs`, `.ome.tif`, `.ome.tiff`, `.jpg`, `.png`, `.ndpi`, `.scn`, `.mrxs`  
- **locs.tsv**: must contain  
  ```
  spot   x   y
  ```
- **cnts.tsv**: genes × spots matrix (tab-delimited).  
- **markers.csv** (optional):  
  ```
  gene,label
  MKI67,Tumor
  KRT20,Mucosa
  ...
  ```

---

## ⚙️ Input Parameters

Parameters are set in the run scripts (`_run_iSCALE_sbatch.sh` or `_run_iSCALE_bsub.sh`).

| Parameter                | Description                                                                 | Default Example            |
|--------------------------|-----------------------------------------------------------------------------|----------------------------|
| `prefix_general`         | Project directory path (must contain `DaughterCaptures` and `MotherImage`)  | `Data/demo/`               |
| `daughterCapture_folders`| List of daughter capture folders                                            | `("D1" "D2" "D3" "D4" D5")`|
| `device`                 | Compute device: `"cuda"` (GPU) or `"cpu"`                                   | `"cuda"`                   |
| `pixel_size_raw`         | Pixel size (µm/pixel) of raw H&E                                            | `0.252`                    |
| `pixel_size`             | Desired pixel size after rescaling                                          | `0.5`                      |
| `n_genes`                | Number of most variable genes to impute                                     | `100`                      |
| `n_clusters`             | Number of clusters for downstream analysis                                  | `20`                       |
| `dist_ST`                | Smoothing parameter across ST captures (integration sharpness)              | `100`                      |

**Notes**  
- `prefix_general` is the main project folder.  
- `dist_ST=100` works well in most cases, but check QC plots in `iSCALE_output/spot_level_st_plots/spots-integrated` to tune if needed.  
- `n_genes=100` is used in the demo because the Xenium dataset has a small targeted panel. For Visium and other platforms with larger gene counts, much higher values (e.g. 3000) are appropriate.

---

## 📤 Output

All results are saved to `iSCALE_output/`:

- **spot_level_st_plots/**  
  QC plots to confirm correct alignment of daughter captures onto mother image.  
- **super_res_gene_expression/**  
  Imputed super-resolution expression (pickle files).  
  - `refined/` subfolder updates predictions for regions unlikely to contain cells.  
- **super_res_ST_plots/**  
  Visualizations of super-resolution gene expression.  
  - includes `refined/`.  
- **clusters-gene_#/**  
  Clustering results using imputed gene expression.  
- **annotation/**  
  Cell-type/region annotations if markers.csv was provided.

---

## 📖 Citation

If you use iSCALE, please cite:

> Schroeder A., et al. *Scaling up spatial transcriptomics for large-sized tissues: uncovering cellular-level tissue architecture beyond conventional platforms.*  
> **Nature Methods** (2025).  
> [https://www.nature.com/articles/s41592-025-02770-8](https://www.nature.com/articles/s41592-025-02770-8)

---

## 📜 License

This project is licensed under the terms of the [LICENSE](./LICENSE.txt) file included in this repository.
