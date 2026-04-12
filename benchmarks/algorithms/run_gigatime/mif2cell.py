import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from pathlib import Path


if __name__ == '__main__':

    mif_file = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/interval/results/luca/gigatime/whole_slide_predictions.pth'
    nucleus_seg_npy_file = '/data1/hounaiqiao/wzr/Simulated_Xenium/luca/cellseg/align_he.npy'
    save_dir = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/interval/results/luca/gigatime/cells/'

    channel_names=[
        'DAPI',
        'TRITC',
        'Cy5',
        'PD-1',
        'CD14',
        'CD4',
        'T-bet',
        'CD34',
        'CD68',
        'CD16',
        'CD11c',
        'CD138',
        'CD20',
        'CD3',
        'CD8',
        'PD-L1',
        'CK',
        'Ki67',
        'Tryptase',
        'Actin-D',
        'Caspase3-D',
        'PHH3-B',
        'Transgelin'
    ]

    # load data
    print('Loading data...')
    mif = torch.load(mif_file)
    nucleus_seg_inst_map = np.load(nucleus_seg_npy_file)

    # assign cell protein expressions
    print('Assigning cell protein expressions...')
    thres = 0.1
    inst_ids = np.unique(nucleus_seg_inst_map)
    inst_ids = inst_ids[inst_ids != 0]
    seg_flat = nucleus_seg_inst_map.ravel()
    mif_flat = mif.reshape(mif.shape[0], -1)
    sizes = np.bincount(seg_flat)
    detected_counts = np.stack([
        np.bincount(seg_flat, weights=mif_flat[c])
        for c in tqdm(range(mif.shape[0]))
    ])
    ratios = detected_counts / sizes.clip(min=1)
    result_bool = (ratios[:, inst_ids] > thres).T
    cells = pd.DataFrame(
        result_bool,
        index=inst_ids,
        columns=channel_names
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cells.to_csv('%s/cells.csv' % save_dir, sep=',', index=True, header=True)

    print('Done!')
