import numpy as np


def get_patches(image: np.ndarray, window_size: int = 512, stride: int = 256) -> tuple:
    """Get sliding window patches.

    Args:
        image (np.ndarray): Image.
        window_size (int): Window size (default is 512).
        stride (int): Stride (default is 256).

    Returns:
        tuple: patches, positions (top left corners), and the original image size.

    """

    H, W = image.shape[:2]

    pad_h = (stride - (H - window_size) % stride) % stride if H > window_size else window_size - H
    pad_w = (stride - (W - window_size) % stride) % stride if W > window_size else window_size - W
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    patches = []
    positions = []  # top left corner

    h = 0
    while h <= padded.shape[0] - window_size:
        w = 0
        while w <= padded.shape[1] - window_size:
            patch = padded[h:h+window_size, w:w+window_size]
            patches.append(patch)

            orig_h = min(h, H - 1)
            orig_w = min(w, W - 1)
            positions.append((orig_h, orig_w))

            w += stride
        h += stride

    return patches, positions, (H, W)
