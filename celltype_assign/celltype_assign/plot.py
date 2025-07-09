import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import json
from pathlib import Path
import tarfile

from matplotlib import cm
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt


def outline(image, contour, col, width):
    background = np.zeros(image.shape[:2], dtype=np.uint8)
    kernel = np.ones((width, width), np.uint8)
    cv2.drawContours(background, contour, -1, 255, -1)
    dilated = cv2.dilate(background, kernel, iterations=1)
    outline = dilated - background
    if len(image.shape) == 2:
        image[outline > 0] = col
    else:
        image[outline > 0, :] = col

    return image


def make_targz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=Path(source_dir).name)

    return


def plot_assigned_celltype(
    image, nuclei_seg, cell_info,
    assigned_colors=None, seg_colors=None, line_width=-1, alpha=None
):
    # default assigned_colors
    if assigned_colors is None:
        all_celltypes = set()
        for v in cell_info.values():
            all_celltypes.add(v['assign'])
            all_celltypes.add(v['gt'])     
        print("All cell types:", all_celltypes)

        all_celltypes = sorted(list(all_celltypes))
        cmap = cm.get_cmap('tab20', len(all_celltypes))
        assigned_colors = {
            ct: tuple(int(255 * c) for c in cmap(i)[:3])
            for i, ct in enumerate(all_celltypes)
        }

    mask_assigned = np.full(image.shape, 255, dtype=np.uint8)
    mask_guessed = np.full(image.shape, 255, dtype=np.uint8)
    mask_gt = np.full(image.shape, 255, dtype=np.uint8)
    mask_seg = np.full(image.shape, 255, dtype=np.uint8)
    if alpha is not None:
        mask_alpha = np.full(image.shape[: 2], 0, dtype=np.uint8)

    for k, v in tqdm(cell_info.items()):
        countour = [np.array(nuclei_seg['nuc'][k]['contour'])]

        cv2.drawContours(mask_assigned, countour, -1, assigned_colors[v['assign']], line_width)
        cv2.drawContours(mask_guessed, countour, -1, assigned_colors[v['random_guess']], line_width)
        cv2.drawContours(mask_gt, countour, -1, assigned_colors[v['gt']], line_width)
        cv2.drawContours(mask_seg, countour, -1, seg_colors[v['type']], line_width)
        if alpha is not None:
            cv2.drawContours(mask_alpha, countour, -1, 255, line_width)

    mask_assigned = cv2.cvtColor(mask_assigned, cv2.COLOR_RGB2RGBA)
    mask_guessed = cv2.cvtColor(mask_guessed, cv2.COLOR_RGB2RGBA)
    mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_RGB2RGBA)
    mask_seg = cv2.cvtColor(mask_seg, cv2.COLOR_RGB2RGBA)
    if alpha is not None:
        mask_assigned[..., 3] = mask_alpha
        mask_guessed[..., 3] = mask_alpha
        mask_gt[..., 3] = mask_alpha
        mask_seg[..., 3] = mask_alpha

    return mask_assigned, mask_guessed, mask_seg


def _plot_color_bar(ax, color_dict, title):
    ax.set_title(title, fontsize=14, fontweight='bold')
    keys = list(color_dict.keys())
    for i, cell_type in enumerate(keys):
        color = tuple(c / 255.0 for c in color_dict[cell_type])
        ax.barh(i, 1, color=color, height=0.8)
        ax.text(0.5, i, cell_type, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(keys) - 0.5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xticks([])
    ax.invert_yaxis()

    return


def create_color_legend(assigned_colors, seg_colors, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    _plot_color_bar(ax1, assigned_colors, 'Assigned Cell Types')
    _plot_color_bar(ax2, seg_colors, 'Segmentation Cell Types')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Color legend saved to %s' % save_path)

    return


if __name__ == '__main__':

    image_file = '/data1/hounaiqiao/yy/NucleiSeg/app/visiumHD/CRC_P2/wsi/crc_p2.tif'
    nuclei_seg_json_file = '/data1/hounaiqiao/yy/NucleiSeg/app/visiumHD/CRC_P2/seg/resized_output.json'
    cell_info_file = '/data1/hounaiqiao/project111/results/visium_hd/cell_info.json'

    save_dir = '/data1/hounaiqiao/project111/results/visium_hd/'

    seg_colors = {
        'Neo': (255, 0, 0),
        'Conn': (0, 255, 0),
        'Inflam': (255, 255, 0),
        'Dead': (0, 255, 255),
        'Epi': (255, 0, 255),
        'unassigned type': (169, 169, 169)
    }

    # assigned_colors = {
    #     "B-cells": (255, 82, 82),
    #     "T-cells": (224, 64, 251),
    #     "Plasmablasts": (255, 64, 129),
    #     "Endothelial": (124, 77, 255),
    #     "PVL": (83, 109, 254),
    #     "Normal Epithelial": (24, 255, 255),
    #     "Cancer Epithelial": (118, 255, 3),
    #     "CAFs": (255, 255, 0),
    #     "Myeloid": (255, 171, 64),
    #     "unassigned type": (169, 169, 169),
    #     "Dead": (169, 169, 169),
    # }

    line_width = -1
    
    # load data
    print('Loading data...')
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    with open(nuclei_seg_json_file, 'r') as f:
        nuclei_seg = json.load(f)
    with open(cell_info_file, 'r') as f:
        cell_info = json.load(f)

    # draw assigned cell type
    print('Drawing assigned cell type...')
    mask_assigned, mask_guessed, mask_seg = plot_assigned_celltype(
        image, nuclei_seg, cell_info, seg_colors=seg_colors,
        line_width = line_width
    )

    # save
    print('Saving...')
    image_resized = cv2.resize(image, fx=0.5, fy=0.5, dsize=(0, 0))
    mask_assigned_resized = cv2.resize(mask_assigned, fx=0.5, fy=0.5, dsize=(0, 0))
    mask_guessed_resized = cv2.resize(mask_guessed, fx=0.5, fy=0.5, dsize=(0, 0))
    # mask_gt_resized = cv2.resize(mask_gt, fx=0.5, fy=0.5, dsize=(0, 0))
    mask_seg_resized = cv2.resize(mask_seg, fx=0.5, fy=0.5, dsize=(0, 0))

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite('%s/image_resized.png' % save_dir, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite('%s/assigned_mask_resized.png' % save_dir, cv2.cvtColor(mask_assigned_resized, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite('%s/guessed_mask_resized.png' % save_dir, cv2.cvtColor(mask_guessed_resized, cv2.COLOR_RGBA2BGRA))
    # cv2.imwrite('%s/gt_mask_resized.png' % save_dir, cv2.cvtColor(mask_gt_resized, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite('%s/seg_mask_resized.png' % save_dir, cv2.cvtColor(mask_seg_resized, cv2.COLOR_RGBA2BGRA))

    make_targz('%s/%s.tar.gz' % (Path(save_dir).parent, Path(save_dir).name), save_dir)

    print('Done!')
