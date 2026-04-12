import argparse
import os
import numpy as np
from utils import load_image_vips, get_image_filename, smart_save_image_vips
from PIL import Image
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None

def adjust_margins_vips(img, pad, pad_value=None):
    h, w = img.height, img.width
    pad_h = (pad - h % pad) % pad
    pad_w = (pad - w % pad) % pad

    if pad_h == 0 and pad_w == 0:
        return img

    top = 0
    bottom = pad_h
    left = 0
    right = pad_w

    if pad_value is None:
        img = img.embed(left, top, w + left + right, h + top + bottom, extend="copy")
    else:
        if isinstance(pad_value, (tuple, list)):
            if len(pad_value) != img.bands:
                raise ValueError("pad_value must match number of channels")
        else:
            pad_value = [pad_value] * img.bands

        img = img.embed(
            left, top, w + left + right, h + top + bottom,
            extend="background", background=pad_value
        )

    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--patchSize', type=int, default=16)
    parser.add_argument('--filename', type=str, default=None,
                        help='Actual filename if known (multi-image mode)')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save the preprocessed image')
    # Compatibility args
    parser.add_argument('--pixelSizeRaw', type=float, default=None)
    parser.add_argument('--pixelSize', type=float, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    pad = args.patchSize ** 2

    print(f"\n ***Preprocessing image...***")

    # Try to use he-scaled if it exists in outputDir
    scaled_candidate = os.path.join(args.outputDir, 'he-scaled')
    try:
        img_path = get_image_filename(scaled_candidate, False)
    except FileNotFoundError:
        # Try in the current directory
        scaled_candidate_local = os.path.join(args.prefix, 'he-scaled')
        try:
            img_path = get_image_filename(scaled_candidate_local, False)
            print(f"✅ Found scaled image in current directory: {img_path}")
        except FileNotFoundError:
            print("⚠️ Scaled image not found in output or current directory — falling back to original.")
            img_path = get_image_filename(os.path.join(args.prefix, 'he-raw'), False)
            print(f"✅ Using original image: {img_path}")

    print(f"Loading image from: {img_path}")

    img = load_image_vips(img_path)
    img = adjust_margins_vips(img, pad=pad, pad_value=255)

    output_dir = args.outputDir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving preprocessed color image to: {output_dir}/he.tiff")
    smart_save_image_vips(img, os.path.join(output_dir, ""), base_name="he", size_threshold=1)

    # === Also save grayscale ===
    #print("Converting image to grayscale...")
    #img_gray = img.colourspace("b-w")   # luminance conversion
    #print(f"Saving grayscale image to: {output_dir}/he_gray.tiff")
    #smart_save_image_vips(img_gray, os.path.join(output_dir, ""), base_name="he_gray", size_threshold=1)


if __name__ == '__main__':
    main()
