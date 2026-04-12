import argparse
import os
from time import time
from PIL import Image
import PIL
import numpy as np
from skimage.transform import rescale

from utils import (
    smart_save_image_vips, load_image_vips, get_image_filename,
    load_image, save_image  
)

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--pixelSizeRaw', type=float, default=None)
    parser.add_argument('--pixelSize', type=float, default=0.5)
    parser.add_argument('--filename', type=str, default=None,
                        help='If provided, use this exact filename instead of the default he-raw.*')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Full path to the output folder for this image')
    return parser.parse_args()


def rescale_image_vips(img, scale):
    if scale == 1.0:
        return img
    img_rescaled = img.resize(scale)
    print(f"After rescaling, image Width: {img_rescaled.width}, image Height: {img_rescaled.height}")
    return img_rescaled 


def rescale_image_numpy(img_np, scale):
    """Fallback: rescale with skimage (slower, memory-heavy, but robust)."""
    if img_np.ndim == 2:
        scale = [scale, scale]
    elif img_np.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError("Unsupported ndim for image array")

    img_rescaled = rescale(img_np, scale, preserve_range=True, anti_aliasing=True)
    return img_rescaled.astype(np.uint8)


def main():
    args = get_args()

    if args.pixelSizeRaw is None:
        raise ValueError("pixelSizeRaw must be provided")

    pixel_size_raw = args.pixelSizeRaw
    pixel_size = args.pixelSize
    scale = pixel_size_raw / pixel_size

    if args.image:
        # Resolve input path
        if args.filename:
            img_dir = os.path.dirname(args.prefix)
            img_path = os.path.join(img_dir, args.filename)
        else:
            img_path = get_image_filename(args.prefix + 'he-raw')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        print(f"\nLoading image from: {img_path}")

        try:
            # ---------- Try pyvips pipeline ----------
            img = load_image_vips(img_path)
            print(f" ***Rescaling image with VIPS (scale: {scale:.3f})...***")
            t0 = time()
            img = rescale_image_vips(img, scale)
            print(f"Rescaling took {int(time() - t0)} sec")
            print(f"Image size: {img.width} x {img.height}")

            output_dir = args.outputDir
            os.makedirs(output_dir, exist_ok=True)

            print("Saving with pyvips...")
            smart_save_image_vips(img, os.path.join(output_dir, ""), base_name="he-scaled")

        except Exception as e:
            # ---------- Fall back to old numpy pipeline ----------
            print(f"⚠️ pyvips failed with error: {e}")
            print("Falling back to numpy/PIL pipeline...")

            img_np = load_image(img_path).astype(np.float32)
            print(f" ***Rescaling image with skimage (scale: {scale:.3f})...***")
            t0 = time()
            img_np = rescale_image_numpy(img_np, scale)
            print(f"Rescaling took {int(time() - t0)} sec")
            print(f"Image shape: {img_np.shape}")

            output_path = os.path.join(args.outputDir, "he-scaled.jpg")
            save_image(img_np, output_path)
            print(f"✅ Saved fallback image to {output_path}")

    


if __name__ == '__main__':
    main()
