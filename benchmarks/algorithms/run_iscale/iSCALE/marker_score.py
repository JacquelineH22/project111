import sys

import numpy as np
from einops import reduce

from utils import read_lines, load_pickle, load_image
from visual import plot_matrix
from scipy import stats



## function to replace the mean score with percentile (fast)
def replace_with_percentile(matrix):
    # Find the percentiles of non-NaN values
    non_nan_values = matrix[~np.isnan(matrix)]
    percentiles = np.percentile(non_nan_values, np.linspace(0, 100, num=100))
    # Replace non-NaN values with their percentiles
    replaced_matrix = np.copy(matrix)
    non_nan_indices = ~np.isnan(matrix)
    values = matrix[non_nan_indices]
    percentile_indices = np.searchsorted(percentiles, values)
    replaced_matrix[non_nan_indices] = percentile_indices / 100.0
    return replaced_matrix



def compute_score_percentile(cnts, mask=None, factor=None, cutoff=0.00001):
    out_cnts = np.full_like(cnts, np.nan, dtype=float)  # start with all NaNs

    for i in range(cnts.shape[2]):
        gene_slice = cnts[:, :, i].astype(float)

        # Apply mask
        if mask is not None:
            gene_slice = np.where(mask, gene_slice, np.nan)

        # Cutoff very small values
        gene_slice[gene_slice < cutoff] = np.nan

        # Apply percentile transform (only to valid values)
        if np.all(np.isnan(gene_slice)):
            out_cnts[:, :, i] = np.nan
        else:
            out_cnts[:, :, i] = replace_with_percentile(gene_slice)

        # Reapply mask so NaNs remain outside tissue
        if mask is not None:
            out_cnts[:, :, i] = np.where(mask, out_cnts[:, :, i], np.nan)

    # Downsample if needed
    if factor is not None:
        out_cnts = reduce(
            out_cnts, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
            h1=factor, w1=factor
        )

    # Normalize each gene channel separately
    for i in range(out_cnts.shape[2]):
        slice_ = out_cnts[:, :, i]
        if np.all(np.isnan(slice_)):
            continue
        min_val, max_val = np.nanmin(slice_), np.nanmax(slice_)
        if max_val > min_val:
            out_cnts[:, :, i] = (slice_ - min_val) / (max_val - min_val + 1e-12)

    # Average across genes, ignore NaNs
    score = np.nanmean(out_cnts, axis=-1)

    # Reapply mask at the very end
    if mask is not None:
        score = np.where(mask, score, np.nan)

    return score



def compute_score(cnts, mask=None, factor=None):
    if mask is not None:
        #cnts = cnts.flatten()

        #mask = mask[:,:,0] #here
        cnts[~mask] = np.nan

    if factor is not None:
        cnts = reduce(
                cnts, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
                h1=factor, w1=factor)

    cnts -= np.nanmin(cnts, (0, 1))
    cnts /= np.nanmax(cnts, (0, 1)) + 1e-12
    score = cnts.mean(-1)

    return score



def get_marker_score(prefix, genes_marker, factor=1):

    genes = read_lines(prefix+'gene-names.txt')
    mask = load_image(prefix+'filterRGB/mask-small-refined.png') > 0

    gene_names = set(genes_marker).intersection(genes)
    cnts = [
            load_pickle(f'{prefix}iSCALE_output/super_res_gene_expression/cnts-super-refined/{gname}.pickle')
            for gname in gene_names]


    #cnts = np.stack(cnts, -1, dtype='float32')
    cnts = np.stack(cnts, -1)
    score = compute_score(cnts, mask=mask, factor=factor)
    return score

def get_marker_score_percentile(prefix, genes_marker, factor=1):

    genes = read_lines(prefix+'gene-names.txt')
    mask = load_image(prefix+'filterRGB/mask-small-refined.png') > 0

    gene_names = set(genes_marker).intersection(genes)
    cnts = [
            load_pickle(f'{prefix}/iSCALE_output/super_res_gene_expression/cnts-super-refined/{gname}.pickle')
            for gname in gene_names]


    #cnts = np.stack(cnts, -1, dtype='float32')
    cnts = np.stack(cnts, -1)
    score = compute_score_percentile(cnts, mask=mask, factor=factor) 
    return score




def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/H123/'
    genes_marker_file = sys.argv[2]  # e.g. 'data/markers/tls.txt'
    outfile = sys.argv[3]  # e.g. 'data/her2st/H123/tls.png'

    # compute marker score
    genes_marker = read_lines(genes_marker_file)
    score = get_marker_score(prefix, genes_marker)

    # visualize marker score
    score = np.clip(
            score, np.nanquantile(score, 0.05),
            np.nanquantile(score, 0.95))
    plot_matrix(score, outfile, white_background=True)


if __name__ == '__main__':
    main()
