import os
import sys
# sys.path.insert(0, "/data1/hounaiqiao/wzr/PanoSpace-main")
import panospace as ps
import scanpy as sc
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import time

os.chdir('/data1/hounaiqiao/wzr/benchmarks/panospace')

##### Load
sample = 'luca'
num_classes=12    # 10 for brca
deconv_method = 'spotiphy'
setting = 'spot100'

Experimental_path = os.path.join('dataset',sample, setting, deconv_method)
seg_path = os.path.join('dataset',sample)
img_dir = f'/data1/hounaiqiao/wzr/Simulated_Xenium/{sample}/HE_aligned//wholeslide/align_he.png'
# segment_adata = sc.read(os.path.join(seg_path,'adata','img_adata_sc.h5ad'))
st_adata = sc.read_h5ad(f'/data1/hounaiqiao/wzr/Simulated_Xenium/{sample}/w55/simulated_square_spot_data.h5ad')
output_dir = os.path.join(Experimental_path,'celltype_infer')
os.makedirs(output_dir, exist_ok=True)

##### cell detector
t0 = time.perf_counter()
if os.path.exists(f'dataset/{sample}/adata_hv/img_adata_sc.h5ad'):
    segment_adata = sc.read_h5ad(f'dataset/{sample}/adata_cellvit/img_adata_sc.h5ad')
    print(segment_adata)
else:
    # Initialize the trainer. Specify pannuke_dir for the dataset location to download and focus for the tissue type to fine-tune on. `focus='all'` mean fine-tune the model with all PanNuke dataset.
    # Available tissue types: 'Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon','Esophagus','HeadNeck',
    # 'Kidney','Liver','Lung','Ovarian','Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus'
    trainer = ps.train_hovernet(pannuke_dir = 'PanNuke', focus='Lung')

    # The `download_pannuke()` method automatically downloads the PanNuke dataset to the specified directory.
    # trainer.download_pannuke()

    # The `split_pannuke()` method divides the raw PanNuke dataset into training and validation sets, converting them into the required format for HoVer-Net.
    trainer.split_pannuke()

    # The `prepare_input()` method processes the PanNuke dataset into a format that HoVer-Net can use for training.
    trainer.prepare_input()

    # The `control_opt()` method updates the path to the pretrained weights in the HoVer-Net’s opt.py file. You can also do this manually if needed.
    trainer.control_opt(hover_net_dir='hover_net')

    # The `control_config()` method adjusts HoVer-Net’s configuration parameters in the config.py file according to your setup. You can also do this manually if needed.
    trainer.control_config(hover_net_dir='hover_net')

    # The `run_train()` method initiates the training process. Note that this will run in a subprocess, meaning no output feedback will be provided until completion. The process can take several hours, depending on your hardware. If you wish to monitor the training progress in real-time, manually run the `run_train.py` script following the HoVer-Net github.
    trainer.run_train()
    segmentor = ps.celldetector(
        img_dir=img_dir,
        tissue_name=sample,
        hover_net_dir='hover_net',
        small_image_size=(1024, 1024),
        resize=None
    )
    segmentor.split_img(cvt=False, hue=None)
    segmentor.run_infer(weight_dir='logs/hovernet_fast_pannuke_type_tf2pytorch.tar')

    segmentor.merge_img()
    segmentor.make_nuclei_adata()
    
    segment_adata = sc.read_h5ad(f'dataset/{sample}/adata/img_adata_sc.h5ad')
t1 = time.perf_counter()
print(f"Elapsed time for segmentation: {t1 - t0:.3f} s")

# exit(0)

label_dict = {
    0 : 'nolabe', 
    1 : 'Neoplastic cells', 
    2 : 'Inflammatory', 
    3 : 'Connective/Soft tissue cells', 
    4 : 'Dead Cells', 
    5 : 'Epithelial' 
}

segment_adata.obs['img_type_str'] = segment_adata.obs['img_type'].map(label_dict)
# fig, ax = plt.subplots(figsize=(5,5), sharey=True)
# sc.pl.spatial(img_adata_sc, color='img_type_str', spot_size=50, ax=ax, colorbar_loc=None)
# plt.savefig(os.path.join(seg_path,'fig','seg.png'), dpi=800, bbox_inches='tight')
# plt.close()

##### Mapping
# Load
t2 = time.perf_counter()
deconv_df = pd.read_csv(f'/data1/hounaiqiao/wzr/benchmarks/spatiocell/deconv_res/{sample}/spot100/{deconv_method}.csv', index_col=0)

deconv_df.index = deconv_df.index.astype(str)
deconv_adata = st_adata.copy()

deconv_df_aligned = deconv_df.reindex(deconv_adata.obs.index)
# deconv_adata.obsm['spatial'] = deconv_adata.obsm['spatial'].astype(int)
deconv_adata.obs = deconv_adata.obs.iloc[:,:2].join(deconv_df_aligned)
deconv_adata.uns['radius'] = int(deconv_adata.uns['spatial']['Xenium_BRCA']['scalefactors']['spot_diameter_fullres'] / 2)
# deconv_adata.uns['radius'] = 55/2
deconv_adata.uns['celltype'] = deconv_df.columns
deconv_adata.obs['cell_count'] = deconv_adata.obs.iloc[:,4:].sum(axis=1)
deconv_adata = deconv_adata[deconv_adata.obs['cell_count'] != 0, :]

sr_adata_name = f'adata/sr_adata_{deconv_method}_spot.h5ad'
sr_deconv_adata_path = os.path.join(Experimental_path, sr_adata_name)

sr_inferencer=ps.DINOv2_superres_deconv(deconv_adata,
                                        deconv_method,
                                        segment_adata,
                                        img_dir,
                                        Experimental_path,
                                        neighb=2,
                                        radius=deconv_adata.uns['radius'],
                                        num_classes=num_classes,
                                        sr_adata_path = sr_adata_name,
                                        device = 'cuda')

if not os.path.exists(os.path.join(Experimental_path,f"sr/superres_model_{deconv_method}_spot.ckpt")):
    sr_inferencer.run_train(devices=[2])

sr_inferencer.run_superres()
sr_deconv_adata = sr_inferencer.sr_adata
print(sr_deconv_adata)

# Celltype assign
inferencer = ps.celltype_annotator.CellTypeAnnotator(experimental_path=Experimental_path,
                               img_dir=img_dir,
                               num_classes=num_classes,
                               deconv_adata=deconv_adata,
                               sr_deconv_adata=sr_deconv_adata,
                               segment_adata=segment_adata,
                               )

inferencer.filter_segmentation()
inferencer.calculate_cell_count()
inferencer.calculate_imgtype_ratio()
inferencer.calculate_celltype_ratio()
# inferencer.calculate_alpha()
# inferencer.calculate_beta()
inferencer.calculate_type_transfer_matrix()
segment_cp = inferencer.infer_cell_types()
print(segment_cp)
segment_cp.write_h5ad(f'dataset/{sample}/{deconv_method}/celltype_infer/adata_cell.h5ad')

t3 = time.perf_counter()
print(f"Elapsed time: {t3 - t2:.3f} s")

# infered_adata = sc.read(f'dataset/{sample}/{deconv_method}/celltype_infer/adata_cell_{deconv_method}.h5ad')
# fig, ax = plt.subplots(figsize=(5,5), sharey=True)
# sc.pl.spatial(infered_adata, cmap='tab10', color='pred_cell_type', spot_size=50, ax=ax, colorbar_loc=None)
# plt.savefig(os.path.join(output_dir,'fig',f'{deconv_method}_pred.png'), dpi=800, bbox_inches='tight')
# plt.close()