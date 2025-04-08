## Set Up Environment

```shell
conda create -n py38 python=3.8 -y
conda activate SpatioCell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements.txt
```

## Nuclei Segmentation and Classification

We developed a novel nuclear segmentation and analysis model that combines a multi-scale feature encoding automatic prompt generator (Prompter) with a fine-tuned SAM model adapted for H&E images. This approach leverages H&E images for precise cell localization and counting, while incorporating cell morphology to generate classification probability profiles.

### Train

The training requires two-step training. Before training, you need to preprocess your dataset.

#### Extract Patches and Preprocess

Before training, extract patches and format data using [extract_patches.py](extract_patches.py) and [preprocess.py](preprocess.py)

```shell
python extract_patches.py
python preprocess.py
```

#### Train SAM

Modified the config files in [configs](configs):
* Set path to the train, valid and test datasets.
* Set number of nuclei types in the training dataset.
* Set path where checkpoints will be saved.
* Set path to the pretrained models.

Run the following command to fintune SAM model.

```shell
CUDA_VISIBLE_DEVICES=0 python train.py configs/type/consep-train-sam.py
```

#### Train Prompter

Run the following command to train the prompter.

```shell
CUDA_VISIBLE_DEVICES=0 python train.py /configs/type/consep-train-prompt.py
```

### Infer

Run inference using the following command.

```shell
CUDA_VISIBLE_DEVICES=0 python test.py \
    configs/type/pannuke-train-prompt.py \
    work_dirs/type/pannuke/train-prompt/latest.pth \
    --format-only --eval-options \
    imgfile_prefix=outs/
```

## Cell Type Assign

To overcome the limitation of multicellular resolution, the cell annotation module of SpatioCell combines H&E-derived and expression-derived information, enabling the construction of high-resolution single-cell spatial maps and improving the accuracy of cell-type assignment in STs.

Run [celltype_assign.py](celltype_assign.py) to annotate cell types in ST data. An example has been uploaded in the [example](example) directory. Plot the results using [plot_assigned_celltype.ipynb](example/plot_assigned_celltype.ipynb) in [example](example).

```shell
python celltype_assign.py
```

The annotation relies on deconvolution-based constraints for cell annotation, but the inherent noise of ST and single-cell reference data—caused by molecular diffusion, sequencing errors, and uneven sampling—make cell-type abundance estimates from deconvolution sometimes deviate from true composition in complex regions.

Here, we introduce Competitive Balance Index (CBI)—a parameter that quantifies the discrepancy between deconvolution results and H&E-derived nuclear classification, to correct errors that arise from deconvolution. Specifically, when CBI exceeds a pre-set threshold, SpatioCell progressively refines the deconvolution result using H&E-derived classification probabilities until the discrepancy falls below the threshold.

Use [celltype_assign_update.py](celltype_assign_update.py) to apply CBI to refine cell type assignment results.
