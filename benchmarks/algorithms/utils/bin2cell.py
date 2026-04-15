import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import anndata as ad
from tqdm import tqdm


def adata2dict(adata: ad.AnnData, x_key: str = 'x', y_key: str = 'y') -> dict:
    """Transform data format from AnnData object to dictionary.

    Args:
        adata (ad.AnnData): AnnData object of bin-level gene expression.
        x_key (str, optional): Key for X coordinate. (default is 'x').
        y_key (str, optional): Key for Y coordinate. (default is 'y).

    Returns:
        dict: Transformed gene expression matrix.

    """
    x_coords = adata.obs[x_key].values
    y_coords = adata.obs[y_key].values

    cols = x_coords // 50 - 1
    rows = y_coords // 50 - 1

    expr_matrix = adata.X

    n_cols = adata.obs[x_key].max() // 50
    n_rows = adata.obs[y_key].max() // 50
    counts_array = np.zeros((len(adata.var_names), n_rows, n_cols), dtype=np.float32)

    for j in range(expr_matrix.shape[1]):
        np.add.at(counts_array[j], (rows, cols), expr_matrix[:, j])

    counts = {
        gene: counts_array[i]
        for i, gene in enumerate(adata.var_names)
    }

    return counts


def bin2cell(counts: dict, nuclei: dict, image: np.ndarray, dilated_pixel: float, method: str = 'overlap', overlap_thres: int = 0.2) -> ad.AnnData:
    """Aggregate bins to cells.

    Args:
        counts (dict): Bin-level gene expression matrix.
        nuclei (dict): Nuclei detection results.
        image (np.ndarray): Full-resolution HE image.
        dilated_pixel (float): Number of pixels to dilate.
        method (str): Aggregating method (default is 'overlap').
        overlap_thres (float): Overlap threshold to consider a bin in a cell (default is 0.2).

    Returns:
        ad.AnnData: AnnData object of cells x genes.

    """
    # dict2array
    nuclei_map = np.zeros(image.shape[: 2], dtype=np.int32)
    for idx in nuclei['nuc'].keys():
        contour = np.array(nuclei['nuc'][idx]['contour'], dtype=np.float32)
        cv2.fillPoly(nuclei_map, [contour.astype(np.int32)], int(idx))

    # nuclei2cells
    cell_map = nuclei_map.copy()
    distance, (iy, ix) = distance_transform_edt(nuclei_map == 0, return_indices=True)
    dilated = (nuclei_map == 0) & (distance <= dilated_pixel)
    cell_map[dilated] = nuclei_map[iy[dilated], ix[dilated]]

    # counts2cells
    cells = {}  # {id: {gene: count}}
    if method == 'overlap':
        n_rows, n_cols = list(counts.values())[0].shape
        scale_factor = round(image.shape[0] / n_rows)
        pbar = tqdm(total=n_rows * n_cols)
        for row in range(n_rows):
            for col in range(n_cols):
                # cells over patch
                y_min = row * scale_factor
                y_max = row * scale_factor + scale_factor
                x_min = col * scale_factor
                x_max = col * scale_factor + scale_factor
                cell_ids, pixel_counts = np.unique(cell_map[y_min: y_max, x_min: x_max], return_counts=True)
                is_cell = cell_ids != 0
                cell_ids = cell_ids[is_cell]
                pixel_counts = pixel_counts[is_cell]
                cell_id = str(cell_ids[np.argmax(pixel_counts)]) if len(cell_ids) > 0 else None
                if len(cell_ids) > 0:
                    if pixel_counts.max() / (scale_factor * scale_factor) > overlap_thres:
                        cell_id = str(cell_ids[np.argmax(pixel_counts)])
                    else:
                        pbar.update()
                        continue
                else:
                    pbar.update()
                    continue

                # aggregate counts
                for gene, matrix in counts.items():
                    count = matrix[row, col]
                    if cell_id not in cells:
                        cells[cell_id] = {}
                    if gene not in cells[cell_id]:
                        cells[cell_id][gene] = 0
                    cells[cell_id][gene] += count

                pbar.update()

    cell_counts = pd.DataFrame.from_dict(cells, orient='index', dtype=np.float32)
    adata = ad.AnnData(X=cell_counts)

    # add cell centroids
    centroids = []
    for i in adata.obs_names:
        centroids.append(nuclei['nuc'][i]['centroid'])
    centroids = np.array(centroids, dtype=np.float32)
    adata.obsm['spatial'] = centroids
    adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']] = centroids

    return adata.copy()


if __name__ == '__main__':
    import scanpy as sc

    from pathlib import Path
    import pickle
    import json

    data_dir = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/istar/results/brca_rep1/istar/cnts-super/'
    nuclei_file = '/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/cellseg_ALL/HE_rep1.json'
    image_file = '/data1/hounaiqiao/wzr/Simulated_Xenium/brca_rep1/HE_aligned/align_he.png'
    save_dir = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/istar/results/brca_rep1/bin2cell/'
    pixel_size = 0.2125
    dilated_pixel = 3 / pixel_size

    # load data
    print('Loading data...')
    counts = {}
    for f in Path(data_dir).glob('*.pickle'):
        with open(f, 'rb') as handle:
            counts[f.stem] = pickle.load(handle)
    with open(nuclei_file, 'r') as f:
        nuclei = json.load(f)
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    # # for h5ad file
    # data_file = '/data1/hounaiqiao/wzr/benchmarks/deconvolution_mapping/result/TESLA/brca_enhanced_exp.h5ad'
    # adata_bin = ad.read_h5ad(data_file)
    # counts = adata2dict(adata_bin, x_key='x', y_key='y')

    # aggregate bins to cells
    print('Aggregating bins to cells...')
    adata = bin2cell(counts, nuclei, image, dilated_pixel, method='overlap')
    adata.obs['batch_name'] = 'brca_rep1'
    sc.pp.calculate_qc_metrics(adata, percent_top=[50, 100, 150, 200], inplace=True, log1p=True)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    adata.write_h5ad('%s/cells.h5ad' % save_dir)

    print('Done!')
