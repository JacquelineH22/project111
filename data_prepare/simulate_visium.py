
import os
import numpy as np
import pandas as pd
import json

import scanpy as sc
import anndata as ad


def Simulated2(adata_cellST, window_size, dataset, outdir):
    """
    Simulate Pseudo-Visium Spots from Single-Cell Spatial ST

    Args:
        adata_cellST : AnnData
            Single-cell spatial AnnData with `.obsm['spatial']` and optional `.obs['annotation']`.
        window_size : int
            Size (in same units as spatial coords) of each pseudo-spot.
        outdir : str
            Directory in which to save outputs.

    Returns:
        None
    """
    # -- prepare output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        print(f"The output file is in {outdir}")

    spatial_df = pd.DataFrame( adata_cellST.obsm['spatial'],
                              index=adata_cellST.obs_names,
                              columns=['coord_X', 'coord_Y']
                              )

    spatial_df['x'] = (spatial_df['coord_X'] // window_size).astype(int)
    spatial_df['y'] = (spatial_df['coord_Y'] // window_size).astype(int)
    spatial_df['spot']  = spatial_df['y'].astype(str) + '_' + spatial_df['x'].astype(str)

    # extract expression matrix as DataFrame: cells × genes
    expr_df = adata_cellST.to_df()

    # aggregate per spot
    agg_expr   = expr_df.groupby(spatial_df['spot']).sum()
    agg_counts = spatial_df.groupby('spot').size().to_frame('cell_count')

    # cell type counts
    if 'annotation' in adata_cellST.obs.columns:
        lbl_df = adata_cellST.obs[['annotation']].join(spatial_df['spot'])
        agg_ctypes = (
            lbl_df.groupby(['spot','annotation'])
                  .size()
                  .unstack(fill_value=0)
        )
        agg_ctypes.to_csv(os.path.join(outdir, 'combined_spot_clusters.tsv'), sep='\t')
    else:
        agg_ctypes = None

    # -- save counts and expression
    agg_counts.to_csv(os.path.join(outdir, 'combined_cell_counts.tsv'), sep='\t')
    agg_expr.to_csv(os.path.join(outdir, 'counts.tsv'), sep='\t')
    print(f"The simulated spot has cell number ranging from {agg_counts['cell_count'].min()} to {agg_counts['cell_count'].max()}")

    # -- compute spot centroid locations in pixel units (requires global `mpp` variable)
    centers = spatial_df.groupby('spot')[['x','y']].first()
    centers['x_pixel'] = (centers['x'] * window_size + window_size/2)
    centers['y_pixel'] = (centers['y'] * window_size + window_size/2)
    centers.to_csv(os.path.join(outdir, 'spot_locations.tsv'), sep='\t')

    # -- build and save spot_info.json
    spot_info = {}
    for spot, idx in spatial_df.groupby('spot').groups.items():
        ids       = idx.tolist()
        centroids = spatial_df.loc[idx, ['coord_X','coord_Y']].apply(tuple, axis=1).tolist()
        labels    = (adata_cellST.obs.loc[idx, 'annotation'].tolist()
                    if 'annotation' in adata_cellST.obs.columns else [None]*len(ids))
        spot_info[spot] = {
            'cell_id': ids,
            'centroid': centroids,
            'annotation': labels
        }
    with open(os.path.join(outdir, 'spot_info.json'), 'w') as f:
        json.dump(spot_info, f, indent=4)

    # assemble final AnnData and save
    adata = ad.AnnData(
        X=agg_expr.values,
        obs=centers[['x','y','x_pixel','y_pixel']].join(agg_ctypes),
        var=adata_cellST.var,
        uns={'spatial':{dataset:{'scalefactors':{'spot_diameter_fullres':window_pxl}}}},
        obsm={'spatial': np.array(centers[['x_pixel','y_pixel']])}
    )
    adata.write(os.path.join(outdir, 'simulated_data.h5ad'))
    print(f"{adata.n_obs} simulated spots")
    
    return adata


if __name__ == "__main__":
    dataset = 'luca'
    st_path = f"/data1/hounaiqiao/wzr/DATA/xenium_data/{dataset}/data/xenium/adata_tacco_anno.h5ad"
    out_dir = f"/data1/hounaiqiao/wzr/Simulated_Xenium/{dataset}/w55_tacco/"
    
    # y_um = 10084.07
    # x_um = 6690.95

    window_size_um = 55
    st_data = sc.read_h5ad(st_path)
    # qc
    if 'high_quality' in st_data.obs.columns:
        st_data = st_data[st_data.obs['high_quality'] == 1, :]
    mpp = round(st_data.uns['H&E resolution'], 5)

    window_pxl = window_size_um / mpp
    st_data.obsm['spatial'] = st_data.obsm['spatial'] / st_data.uns['H&E resolution']
    # y = y_um / mpp
    # x = x_um / mpp
    
    spot_adata = Simulated2(st_data, window_pxl, dataset, out_dir)
    print(spot_adata)

