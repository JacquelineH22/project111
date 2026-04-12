#!/usr/bin/env python3
import argparse
import os
import numpy as np
from PIL import Image
from skimage import filters
from utils import load_image, get_image_filename
from extract_features import patchify
import pickle

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True,
                        help="Path prefix for the sample (must contain he.jpg and mask.png)")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Superpixel size (default=16)")
    args = parser.parse_args()

    prefix = args.prefix
    patch_size = args.patch_size

    # --- Load H&E image ---
    img_path = get_image_filename(args.prefix + 'he')
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")

    print(f"\nLoading image from: {img_path}")
    he_np = load_image(img_path)

    # convert to numpy for patchify
    #he_np = np.ndarray(
    #    buffer=he.write_to_memory(),
    #    dtype=np.uint8,
    #    shape=[he.height, he.width, he.bands]
    #)

    # --- Load mask and downsample to superpixel grid ---
    mask_orig = np.array(Image.open(os.path.join(prefix, "mask.png")).convert("L"))
    mask_bin = mask_orig > 0

    super_y = he_np.shape[0] // patch_size
    super_x = he_np.shape[1] // patch_size

    # crop mask to multiple of patch_size
    mask_cropped = mask_bin[:super_y * patch_size, :super_x * patch_size]
    mask_super = mask_cropped.reshape(super_y, patch_size, super_x, patch_size).any(axis=(1, 3))

    # --- Patchify into superpixels ---
    he_tiles, shapes = patchify(he_np, patch_size=patch_size)

    he_mean = np.zeros(len(he_tiles), dtype=np.float32)
    he_std = np.zeros(len(he_tiles), dtype=np.float32)

    for idx, tile in enumerate(he_tiles):
        i, j = divmod(idx, super_x)
        if mask_super[i, j]:
            he_mean[idx] = np.mean(tile)
            he_std[idx] = np.std(tile)
        else:
            he_mean[idx] = 0.0
            he_std[idx] = 0.0

    # --- Normalize mean and std ---
    mean_norm = (he_mean - he_mean.min()) / (he_mean.max() - he_mean.min() + 1e-8)
    std_norm = (he_std - he_std.min()) / (he_std.max() - he_std.min() + 1e-8)

    # --- Ratio ---
    ratio = std_norm / (mean_norm + 1e-8)

    # --- Remove top 5% before Otsu ---
    cutoff = np.percentile(ratio, 95)
    ratio_filtered = ratio[ratio <= cutoff]

    # Otsu threshold
    otsu_thresh = filters.threshold_otsu(ratio_filtered)
    conserve_index = ratio >= otsu_thresh  # keep "high score" regions

    # Save conserve index at superpixel level
    os.makedirs(os.path.join(prefix, "filterRGB"), exist_ok=True)
    save_pickle(conserve_index, os.path.join(prefix, "filterRGB/conserve_index.pickle"))

    # --- Build superpixel mask (mask-small) ---
    super_grid = np.array(conserve_index).reshape((super_y, super_x))

    mask_small = np.zeros((super_y, super_x), dtype=np.uint8)
    mask_small[super_grid] = 255
    Image.fromarray(mask_small).save(os.path.join(prefix, "filterRGB/mask-small-refined.png"))

    # --- Build pixel-level mask ---
    mask_full = np.zeros((he_np.shape[0], he_np.shape[1]), dtype=np.uint8)
    for i in range(super_y):
        for j in range(super_x):
            if super_grid[i, j]:
                mask_full[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size] = 255

    # Combine with original mask
    combined_mask = np.where((mask_full == 0) | (mask_orig == 0), 0, 255).astype(np.uint8)
    Image.fromarray(combined_mask).save(os.path.join(prefix, "filterRGB/mask-refined.png"))

    print(f"Saved refined masks to {prefix}")

if __name__ == "__main__":
    main()
