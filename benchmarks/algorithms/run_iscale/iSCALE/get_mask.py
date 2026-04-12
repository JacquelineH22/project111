#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image

from utils import save_image, load_pickle, cluster_mask
from connected_components import relabel_small_connected
from image import crop_image


def remove_margins(embs, mar):
    for ke, va in embs.items():
        embs[ke] = [
            v[mar[0][0]:-mar[0][1], mar[1][0]:-mar[1][1]]
            for v in va]


def get_mask_embeddings(embs, mar=16, min_connected=4000):
    n_clusters = 2

    # remove margins to avoid border effects
    remove_margins(embs, ((mar, mar), (mar, mar)))

    # concatenate features
    x = np.concatenate(list(embs.values()))

    # segment into 2 clusters
    labels, __ = cluster_mask(x, n_clusters=n_clusters, method='km')
    labels = relabel_small_connected(labels, min_size=min_connected)

    # choose cluster with higher RGB variance as tissue foreground
    rgb = np.stack(embs['rgb'], -1)
    i_foreground = np.argmax([
        rgb[labels == i].std() for i in range(n_clusters)])
    mask_small = (labels == i_foreground).astype(np.uint8)

    # restore margins
    extent = [(-mar, s + mar) for s in mask_small.shape]
    mask_small = crop_image(
        mask_small, extent,
        mode='constant', constant_values=mask_small.min())

    return mask_small


def upscale_to_full(mask_small, patch_size=16):
    """Upscale superpixel mask to full resolution."""
    h_small, w_small = mask_small.shape
    h_full, w_full = h_small * patch_size, w_small * patch_size
    mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    for i in range(h_small):
        for j in range(w_small):
            value = 255 if mask_small[i, j] else 0
            mask_full[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ] = value
    return mask_full


def main():
    if len(sys.argv) < 3:
        print("Usage: python get_mask.py <input_embeddings.pkl> <output_prefix>")
        sys.exit(1)

    inpfile = sys.argv[1]
    outprefix = sys.argv[2]

    embs = load_pickle(inpfile)
    mask_small = get_mask_embeddings(embs)

    # Save small (superpixel) mask
    mask_small_img = Image.fromarray(mask_small * 255)
    mask_small_img.save(outprefix + "mask-small.png")

    # Save full-resolution pixel-level mask
    mask_full = upscale_to_full(mask_small, patch_size=16)
    mask_full_img = Image.fromarray(mask_full)
    mask_full_img.save(outprefix + "mask.png")

    print(f"Saved {outprefix}mask-small.png and {outprefix}mask.png")


if __name__ == '__main__':
    main()
