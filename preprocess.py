from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from pathlib import Path


if __name__ == '__main__':
    train_root = './datasets/type/pannuke/patches/fold1/384x384_128x128/'
    valid_root = './datasets/type/pannuke/patches/fold2/384x384_128x128/'
    with_type = True

    save_root = './datasets/type/pannuke/preprocessed/'
    for d1 in ['images', 'labels']:
        for d2 in ['train', 'valid']:
            Path('%s/%s/%s' % (save_root, d1, d2)).mkdir(parents=True, exist_ok=True)
    
    roots = {'train': Path(train_root), 'valid': Path(valid_root)}

    for mode in ['train', 'valid']:
        root = roots[mode]
        for file in tqdm(list(root.glob('*.npy'))):
            img_label = np.load(file)
            img = img_label[..., :3].astype(np.uint8)
            img_rgb = np.stack([img, img, img], axis=-1)
            label = img_label[..., 3]
            if with_type:
                assert img_label.shape[-1] > 4, 'No type label given in ndarray.'
                cls_label = img_label[...,4]
                label_dict = {'inst_map': label, 'type_map': cls_label}
                torch.save(label_dict, '%s/labels/%s/%s.pth' % (save_root, mode, file.stem))
            else:
                np.save('%s/labels/%s/%s.npy' % (save_root, mode, file.stem), label)

            Image.fromarray(img).save('%s/images/%s/%s.png' % (save_root, mode, file.stem))

    print('Done!')
