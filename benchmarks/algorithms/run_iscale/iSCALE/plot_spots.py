import sys

import numpy as np
from einops import reduce

from utils import load_image, load_tsv, read_lines, read_string
from visual import plot_spots
import matplotlib



def plot_spots_multi(
        cnts, locs, gene_names, radius, img, prefix):
    for i, gname in enumerate(gene_names):
        ct = cnts[:, i]
        outfile = f'{prefix}{gname}.png'
        plot_spots(
                img=img, cnts=ct, locs=locs, radius=radius,
                cmap='turbo', weight=0.8, 
                outfile=outfile)


def main():
    prefix = sys.argv[1]  
    #grayHE_flag = sys.argv[2].lower() in ("true", "1", "yes")
    arg = sys.argv[2]
    if "=" in arg:
        key, val = arg.split("=")
        grayHE_flag = val.lower() in ("true","1","yes")
    else:
        grayHE_flag = arg.lower() in ("true","1","yes")

    factor = 16
    print(grayHE_flag)

    infile_cnts = f'{prefix}cnts.tsv'
    infile_locs = f'{prefix}locs.tsv'
    infile_img = f'{prefix}he.tiff'
    infile_genes = f'{prefix}gene-names.txt'
    infile_radius = f'{prefix}radius.txt'

    # load data
    cnts = load_tsv(infile_cnts)
    locs = load_tsv(infile_locs)

    print(cnts.index)
    print(locs.index)

    assert (cnts.index == locs.index).all()
    spot_radius = int(read_string(infile_radius))
    img = load_image(infile_img)

    if grayHE_flag:
        if img.ndim == 3 and img.shape[2] == 3:
            # Convert RGB H&E to grayscale
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            # Stack back into 3 channels so overlay still works
            img = np.repeat(gray[..., np.newaxis], 3, axis=-1)

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    #if img.ndim == 2:
    #    img = np.tile(img[..., np.newaxis], 3)

    # select genes
    gene_names = read_lines(infile_genes)
    cnts = cnts[gene_names]
    cnts = cnts.to_numpy()

    # recale image
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= factor
    locs = locs.round().astype(int)
    img = reduce(
            img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)

    # rescale spot
    spot_radius = np.round(spot_radius / factor).astype(int)

    # plot spot-level gene expression measurements
    plot_spots_multi(
            cnts=cnts,
            locs=locs, gene_names=gene_names,
            radius=spot_radius, 
            img=img, prefix=prefix+'iSCALE_output/spot_level_ST_plots/spots/')


if __name__ == '__main__':
    main()
