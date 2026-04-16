#!/usr/bin/env python
"""
Split a full Xenium pseudo-Visium dataset into multiple daughter captures,
mimicking the Xenium benchmarking setup described in the iSCALE paper.

Paper setup:
- Mother slide: full Xenium H&E (12mm x 24mm)
- Daughter captures: multiple 3.2mm x 3.2mm regions (D1, D2, ...)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil


def visualize_grid(locs, grid_info, output_dir):
    """Plot the spatial grid of daughter captures and save to PNG."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(locs['x'], locs['y'], c='lightgray', s=1, alpha=0.5, label='All spots')

    colors = plt.cm.tab20(np.linspace(0, 1, len(grid_info)))
    for i, (name, info) in enumerate(grid_info.items()):
        x_min, x_max = info['x_range']
        y_min, y_max = info['y_range']
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              linewidth=2, edgecolor=colors[i],
                              facecolor=colors[i], alpha=0.2)
        ax.add_patch(rect)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        ax.text(cx, cy, f"{name}\n{info['n_spots']} spots",
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Daughter Captures Grid Layout', fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'captures_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved: {out_path}")


def split_xenium_data(h5ad_path, he_path, output_base,
                      capture_size_mm=3.2, pixel_scale=0.2125,
                      overlap_mm=0.0, min_spots=100):
    """
    Split Xenium data into multiple daughter captures.

    Args:
        h5ad_path: Path to input h5ad file.
        he_path: Path to H&E image.
        output_base: Root output directory.
        capture_size_mm: Capture tile size in mm.
        pixel_scale: Pixel size in µm/pixel.
        overlap_mm: Overlap between adjacent captures in mm.
        min_spots: Minimum spots required to keep a capture.
    """
    import anndata as ad

    capture_size_px = capture_size_mm * 1000 / pixel_scale
    overlap_px = overlap_mm * 1000 / pixel_scale
    step_px = capture_size_px - overlap_px

    # Load data
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.shape[0]:,} spots x {adata.shape[1]} genes")

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    cnts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    locs = adata.obs[['x_pixel', 'y_pixel']].copy()
    locs.columns = ['x', 'y']

    x_min, x_max = locs['x'].min(), locs['x'].max()
    y_min, y_max = locs['y'].min(), locs['y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min

    n_x = int((x_span - overlap_px) / step_px)
    n_y = int((y_span - overlap_px) / step_px)
    print(f"Grid: {n_x} x {n_y} = {n_x * n_y} captures "
          f"({capture_size_mm}mm tiles, {pixel_scale} µm/px)")

    if n_x * n_y == 0:
        print(f"WARNING: Data extent too small for {capture_size_mm}mm captures. "
              "Consider reducing capture_size_mm.")
        return

    # Build capture grid
    grid_info = {}
    capture_id = 1
    for i in range(n_x):
        for j in range(n_y):
            cap_x_min = x_min + i * step_px
            cap_x_max = cap_x_min + capture_size_px
            cap_y_min = y_min + j * step_px
            cap_y_max = cap_y_min + capture_size_px

            mask = (
                (locs['x'] >= cap_x_min) & (locs['x'] < cap_x_max) &
                (locs['y'] >= cap_y_min) & (locs['y'] < cap_y_max)
            )
            n = mask.sum()
            if n < min_spots:
                continue

            locs_cap = locs[mask].copy()
            name = f"D{capture_id}"
            grid_info[name] = {
                'grid_pos': (i, j),
                'x_range': (cap_x_min, cap_x_max),
                'y_range': (cap_y_min, cap_y_max),
                'n_spots': n,
                'locs': locs_cap,
                'cnts': cnts.loc[locs_cap.index].copy(),
            }
            capture_id += 1

    if not grid_info:
        print("ERROR: No valid captures generated.")
        return

    # Save outputs
    aligned_dir = os.path.join(output_base, 'DaughterCaptures/AllignedToMother')
    mother_dir = os.path.join(output_base, 'MotherImage')
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(mother_dir, exist_ok=True)

    for name, info in grid_info.items():
        cap_dir = os.path.join(aligned_dir, name)
        os.makedirs(cap_dir, exist_ok=True)
        info['cnts'].to_csv(os.path.join(cap_dir, 'cnts.tsv'), sep='\t')
        info['locs'].to_csv(os.path.join(cap_dir, 'locs.tsv'), sep='\t')

    shutil.copy2(he_path, os.path.join(mother_dir, 'he-raw.png'))

    radius_raw = (55 / 2) / pixel_scale  # spot window radius in pixels (55µm diameter)
    with open(os.path.join(mother_dir, 'radius-raw.txt'), 'w') as f:
        f.write(f"{radius_raw:.2f}\n")

    visualize_grid(locs, grid_info, output_base)

    # Summary
    total_spots = sum(info['n_spots'] for info in grid_info.values())
    print(f"\nDone. {len(grid_info)} captures, {total_spots:,} / {len(locs):,} spots "
          f"({total_spots / len(locs) * 100:.1f}% coverage)")
    print(f"Output: {output_base}")


if __name__ == '__main__':
    H5AD_PATH = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/SC/simulated_data.h5ad'
    HE_PATH   = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/HE/align_he.png'
    OUTPUT_BASE = '/data1/linxin/Benchmark/iSCALE/iSCALE/benchmark_data/iscale_input_multi_captures'

    CAPTURE_SIZE_MM = 3.2    # tile size in mm (paper uses 3.2mm)
    PIXEL_SCALE     = 0.2125 # Xenium pixel size in µm/pixel
    OVERLAP_MM      = 0.0    # overlap between tiles in mm
    MIN_SPOTS       = 100    # discard captures with fewer spots than this

    split_xenium_data(
        h5ad_path=H5AD_PATH,
        he_path=HE_PATH,
        output_base=OUTPUT_BASE,
        capture_size_mm=CAPTURE_SIZE_MM,
        pixel_scale=PIXEL_SCALE,
        overlap_mm=OVERLAP_MM,
        min_spots=MIN_SPOTS,
    )
