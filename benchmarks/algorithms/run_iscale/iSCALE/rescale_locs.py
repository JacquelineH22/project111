import argparse
import os
from time import time
from PIL import Image
import PIL

from skimage.transform import rescale
import numpy as np

from utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv)

Image.MAX_IMAGE_PIXELS = None

PIL.Image.MAX_IMAGE_PIXELS = 10e100


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


# def rescale_image(img, scale):
#     if img.ndim == 2:
#         img = rescale(img, scale, preserve_range=True)
#     elif img.ndim == 3:
#         channels = img.transpose(2, 0, 1)
#         channels = [rescale_image(c, scale) for c in channels]
#         img = np.stack(channels, -1)
#     else:
#         raise ValueError('Unrecognized image ndim')
#     return img


def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--pixelSizeRaw', type=float, default=None)
    parser.add_argument('--pixelSize', type=float, default=0.5)
    parser.add_argument('--locs', action='store_true')
    parser.add_argument('--radius', action='store_true')


    args = parser.parse_args()
    return args


def main():
    

    args = get_args()

    scale = args.pxl_size_raw / args.pxl_size


    if args.locs:
        locs = load_tsv(args.prefix+'locs-raw.tsv')
        locs = locs * scale
        locs = locs.round().astype(int)
        save_tsv(locs, args.prefix+'locs.tsv')

    if args.radius:
        radius = float(read_string(args.prefix+'radius-raw.txt'))
        radius = radius * scale
        radius = np.round(radius).astype(int)
        write_string(radius, args.prefix+'radius.txt')


if __name__ == '__main__':
    main()
