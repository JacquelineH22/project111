import os
import sys
# sys.path.insert(0, '/data1/hounaiqiao/wzr/Spotiphy')
import spotiphy

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import scanpy as sc
import torch
import cv2
import tifffile as tiff

import argparse
import json


def deconvolution(adata_sc, adata_st, results_folder, key_type='label'):
    # preprocess
    type_list = sorted(list(adata_sc.obs[key_type].unique().astype(str)))
    print(f'There are {len(type_list)} cell types: {type_list}')
    adata_st.var_names_make_unique()
    adata_sc, adata_st = spotiphy.initialization(adata_sc, adata_st, verbose=1)

    # marker gene selection
    marker_gene_dict = spotiphy.sc_reference.marker_selection(adata_sc, key_type=key_type, return_dict=True, 
                                                          n_select=50, threshold_p=0.1, threshold_fold=1.2,
                                                          q=0.12)
    marker_gene = []
    marker_gene_label = []
    for type_ in type_list:
        marker_gene.extend(marker_gene_dict[type_])
        marker_gene_label.extend([type_]*len(marker_gene_dict[type_]))
    marker_gene_df = pd.DataFrame({'gene':marker_gene, 'label':marker_gene_label})
    marker_gene_df.to_csv(results_folder+'marker_gene.csv')
    # Filter scRNA and spatial matrices with marker genes
    adata_sc_marker = adata_sc[:, marker_gene]
    adata_st_marker = adata_st[:, marker_gene]

    # construct reference
    sc_ref = spotiphy.construct_sc_ref(adata_sc_marker, key_type=key_type)
    spotiphy.sc_reference.plot_heatmap(adata_sc_marker, key_type, save=True, out_dir=results_folder)

    # cell proportion estimation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.array(adata_st_marker.X)
    X = np.nan_to_num(X, nan=0.0)
    
    cell_proportion = spotiphy.deconvolution.estimation_proportion(X, adata_sc_marker, sc_ref, type_list, key_type, n_epoch=8000,
                                                            plot=True, batch_prior=1, device=device)
    adata_st.obs[type_list] = cell_proportion
    np.save(results_folder+'proportion.npy', cell_proportion)
    adata_st.obs[type_list].to_csv(results_folder+'prop_spotiphy.csv')

    return cell_proportion


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Spotiphy Benchmark Script")
    parser.add_argument('--dataset', type=str, default='brca_rep1', help='dataset')
    parser.add_argument('--ws', type=int, default=55, help='window size')
    parser.add_argument('--scfile', type=str, required=True, help='filepath for SC data')
    parser.add_argument('--spot', action = 'store_true')
    args = parser.parse_args()

    # set input and output dir
    data_folder = f'/data1/hounaiqiao/wzr/Simulated_Xenium/{args.dataset}/w{args.ws}/'
    if args.spot:
        results_folder = f'../result/spotiphy/{args.dataset}/spot{args.ws}/'
    else:
        results_folder = f'../result/spotiphy/{args.dataset}/ws{args.ws}/'
    # results_folder = '/data1/hounaiqiao/yy/NucleiSeg/app/visium/10x_ovary/assign_spotiphy_nc/'
    if not os.path.exists(results_folder):
        # Create result folder if it does not exist
        os.makedirs(results_folder)
    
    # load data
    if args.spot:
        adata_st_orig = sc.read_h5ad(data_folder + 'simulated_square_spot_data.h5ad')
    else:
        adata_st_orig = sc.read_h5ad(data_folder + 'simulated_data.h5ad')
    
    print(adata_st_orig.obs.columns)
    # adata_st = sc.read_h5ad('/data1/hounaiqiao/yy/NucleiSeg/app/visium/10x_ovary/data/ov_visium.h5ad')
    adata_st_orig.var_names_make_unique()
    adata_st_orig.obsm['spatial'] = np.array(adata_st_orig.obs[['x_pixel', 'y_pixel']], dtype=np.int32)
    adata_st = adata_st_orig.copy()
    
    adata_sc_orig = sc.read_h5ad(args.scfile)
    adata_sc_orig.var_names_make_unique()
    adata_sc = adata_sc_orig.copy()
    
    if args.dataset == 'brca_rep1':
        pixel_size = 0.2125  # microns per pixel
        spot_radius = args.ws/(2*pixel_size)
    elif args.dataset == 'Xenium_5k_COAD':
        pixel_size = 1
        spot_radius = adata_st.uns['spatial']['Xenium_5k_COAD']['scalefactors']['spot_diameter_fullres'] / 2
    
    # prepare valid
    print(adata_sc.obs.columns)
    key_type = 'label'  # 'major_annotation for COAD', 'Cell Annotation' for ovca, 'label' for brca
    type_list = sorted(list(adata_sc.obs[key_type].unique().astype(str)))

    # deconvolution
    if not os.path.exists(results_folder+'proportion.npy'):
        cell_proportion = deconvolution(adata_sc, adata_st, results_folder, key_type)
    else:
        cell_proportion = np.load(results_folder+'proportion.npy')
    
    # segmentation
    segment_dir = f"/data1/hounaiqiao/wzr/benchmarks/deconvolution_mapping/result/spotiphy/{args.dataset}/segmentation/"
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    print('Load image data...')  
    if args.dataset == 'brca_rep1':
        img = cv2.imread(f'/data1/hounaiqiao/wzr/Simulated_Xenium/{args.dataset}/HE_aligned/align_he.png')
    elif args.dataset == 'Xenium_5k_COAD':
        img = tiff.imread('/data1/hounaiqiao/wzr/Simulated_Xenium/Xenium_5k_COAD/HE_aligned/align_he.tif')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(f'/data1/hounaiqiao/wzr/Simulated_Xenium/{args.dataset}/HE_aligned/wholeslide/align_he_scale.jpg')

    if not os.path.exists(segment_dir + "n_cell_per_spot.csv"):
        print("Segmentation Starting...")
        segmentation = spotiphy.segmentation.Segmentation(
            img[:, :, [2, 1, 0]],
            adata_st.obsm["spatial"],
            n_tiles=(4, 4, 1),
            spot_shape="square",
            spot_radius= spot_radius,
            out_dir=segment_dir,
            enhancement=True,
        )
        segmentation.segment_nucleus(save=True)
        n_cell_df = segmentation.n_cell_df
        # Save the segmentation for future usage
        df = pd.DataFrame(n_cell_df)
        df.to_csv(segment_dir + "n_cell_per_spot.csv", index=False)
    else:
        print('Load segmentation data...')
        segmentation = spotiphy.segmentation.Segmentation(
            np.zeros((10,10,3), dtype=np.uint8),    # 后续用不到img
            adata_st.obsm["spatial"],
            n_tiles=(4, 4, 1),
            spot_shape="square",
            spot_radius=spot_radius,
            out_dir=segment_dir,
            enhancement=True,
        )
        # boundary = np.load(f"{results_folder}segmentation/segmentation_boundary.npy")
        # segmentation.nucleus_boundary = boundary
        # segmentation.label = np.load(f"{results_folder}segmentation/segmentation_label.npy")
        # segmentation.probability = np.load(f"{results_folder}segmentation/segmentation_probability.npy")
        nucleus_df_array = np.load(segment_dir + "nucleus_df.csv.npy", allow_pickle=True)
        nucleus_df = pd.DataFrame(nucleus_df_array, columns=['x', 'y', 'in_spot'])
        segmentation.nucleus_df = nucleus_df
        
        segmentation.n_cell_df = segmentation.n_cell_in_spot(segmentation.nucleus_df[['x', 'y']].values, 
                                                             segmentation.spot_center, segmentation.spot_radius,
                                                             segmentation.spot_shape, segmentation.nucleus_df)
        
        
        # segmentation.n_cell_df = pd.read_csv(f"{results_folder}segmentation/n_cell_per_spot.csv")
        segmentation.is_segmented = True

    # decomposition
    print('Start decomposition...')
    n_cell = segmentation.n_cell_df["cell_count"].values
    # adata_st_decomposed = spotiphy.deconvolution.decomposition(
    #     adata_st_orig,
    #     adata_sc_orig,
    #     key_type,
    #     cell_proportion,
    #     save=True,
    #     out_dir=results_folder,
    #     verbose=1,
    #     spot_location=adata_st_orig.obsm["spatial"],
    #     filtering_gene=True,
    #     n_cell=n_cell,
    #     filename="ST_decomposition.h5ad",
    # )
    
    """
    # for visualization
    search_direction = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [2, 0],
        [0, 2],
        [-2, 0],
        [0, -2],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ]
    boundary_dict = spotiphy.segmentation.cell_boundary(
        segmentation.nucleus_df[["x", "y"]].values,
        img_size=img.shape[:2],
        max_dist=25,
        max_area=580,
        verbose=0,
        search_direction=search_direction,
        delta=2,
    )
    # Save the dictionary of the cell boundaries for future usage
    with open(results_folder + f"segmentation/boundary_dict.pkl", "wb") as file:
        pickle.dump(boundary_dict, file)
    """
    
    cell_number = spotiphy.deconvolution.proportion_to_count(
        cell_proportion, segmentation.n_cell_df["cell_count"].values, multiple_spots=True
    )
    
    print('Assign celltype in spots...')
    segmentation.nucleus_df = spotiphy.deconvolution.assign_type_spot(
        segmentation.nucleus_df, segmentation.n_cell_df, cell_number, type_list
    )
    
    print('Assign celltype outside spots...')
    segmentation.nucleus_df, cell_proportion_smooth = (
    spotiphy.deconvolution.assign_type_out(
            segmentation.nucleus_df,
            cell_proportion,
            segmentation.spot_center,
            type_list,
            max_distance = 110 / pixel_size,
            band_width=150,
        )
    )

    print(segmentation.nucleus_df)
    print(type(segmentation))
    segmentation.nucleus_df.to_csv(Path(results_folder,'nucleus_df_slide.csv'), index=False)
    
    # plot_visium = spotiphy.plot.Plot_Visium(
    #     segmentation=segmentation, boundary_dict=boundary_dict, type_list=type_list
    # )
    # plot_visium.plot_legend(save=f"{results_folder}legend.png")