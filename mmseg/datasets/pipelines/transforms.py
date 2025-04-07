# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
import math
import colorsys
from PIL import Image
import inspect
import copy
import random
from matplotlib import pyplot as plt
from pathlib import Path

from scipy.ndimage import maximum_filter1d, find_objects
from numba import njit, float32, int32, vectorize
import mmcv
import torch
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from scipy import ndimage
import scipy
import fastremap
from skimage import morphology as morph
from skimage.transform import warp, ProjectiveTransform
import cv2

from ..builder import PIPELINES
from ...utils import box_noise

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(
        self, img_scale=None, multiscale_mode="range", ratio_range=None, keep_ratio=True
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results["img"].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range
                )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results["img"], results["scale"], return_scale=True
            )
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results["img"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results["img"], results["scale"], return_scale=True
            )
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape  # in case that there is no padding
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results["scale"], interpolation="nearest"
                )
            else:
                gt_seg = mmcv.imresize(
                    results[key], results["scale"], interpolation="nearest"
                )
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(img_scale={self.img_scale}, "
            f"multiscale_mode={self.multiscale_mode}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({"flip_ratio": "prob"}, cls_name="RandomFlip")
    def __init__(self, prob=None, direction="horizontal"):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ["horizontal", "vertical"]

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if "flip" not in results:
            flip = True if np.random.rand() < self.prob else False
            results["flip"] = flip
        if "flip_direction" not in results:
            results["flip_direction"] = self.direction
        if results["flip"]:
            # flip image
            results["img"] = mmcv.imflip(
                results["img"], direction=results["flip_direction"]
            )

            # flip segs
            for key in results.get("seg_fields", []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                ).copy()
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0, padding_mode='constant', seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.padding_mode = padding_mode
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results["img"], shape=self.size, pad_val=self.pad_val, padding_mode=self.padding_mode
            )
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results["img"], self.size_divisor, pad_val=self.pad_val
            )
        results["img"] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get("seg_fields", []):
            if key == "gt_instance":
                results[key] = mmcv.impad(
                    results[key], shape=results["pad_shape"][:2], pad_val=0
                )
            else:
                results[key] = mmcv.impad(
                    results[key],
                    shape=results["pad_shape"][:2],
                    pad_val=self.seg_pad_val,
                )

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(size={self.size}, size_divisor={self.size_divisor}, "
            f"pad_val={self.pad_val})"
        )
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results["img"] = mmcv.imnormalize(
            results["img"], self.mean, self.std, self.to_rgb
        )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class Rerange(object):
    """Rerange the image pixel value.

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results["img"]
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results["img"] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(min_value={self.min_value}, max_value={self.max_value})"
        return repr_str


@PIPELINES.register_module()
class CLAHE(object):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results["img"].shape[2]):
            results["img"][:, :, i] = mmcv.clahe(
                np.array(results["img"][:, :, i], dtype=np.uint8),
                self.clip_limit,
                self.tile_grid_size,
            )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(clip_limit={self.clip_limit}, " f"tile_grid_size={self.tile_grid_size})"
        )
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results["gt_semantic_seg"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(
        self, prob, degree, pad_val=0, seg_pad_val=255, center=None, auto_bound=False
    ):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f"degree {degree} should be positive"
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, (
            f"degree {self.degree} should be a " f"tuple of (min, max)"
        )
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results["img"] = mmcv.imrotate(
                results["img"],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound,
            )

            # rotate segs
            for key in results.get("seg_fields", []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(prob={self.prob}, "
            f"degree={self.degree}, "
            f"pad_val={self.pal_val}, "
            f"seg_pad_val={self.seg_pad_val}, "
            f"center={self.center}, "
            f"auto_bound={self.auto_bound})"
        )
        return repr_str


@PIPELINES.register_module()
class RGB2Gray(object):
    """Convert RGB image to grayscale image.

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def __call__(self, results):
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results["img"]
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results["img"] = img
        results["img_shape"] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(out_channels={self.out_channels}, " f"weights={self.weights})"
        return repr_str


@PIPELINES.register_module()
class AdjustGamma(object):
    """Using gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]
        ).astype("uint8")

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results["img"] = mmcv.lut_transform(
            np.array(results["img"], dtype=np.uint8), self.table
        )

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(gamma={self.gamma})"


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get("seg_fields", []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation="nearest"
                )
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(scale_factor={self.scale_factor})"


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta, self.brightness_delta),
            )
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img, alpha=np.uniform(self.contrast_lower, self.contrast_upper)
            )
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_lower, self.saturation_upper),
            )
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (
                img[:, :, 0].astype(int)
                + np.random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results["img"]
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@PIPELINES.register_module()
class GetHVMap(object):
    def __init__(self):
        super(GetHVMap, self).__init__()

    def gen_instance_hv_map(self, ann):
        """Input annotation must be of original shape.

        The map is calculated only for instances within the crop portion
        but based on the original shape in original image.

        Perform following operation:
        Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        """

        def fix_mirror_padding(ann):
            """Deal with duplicated instances due to mirroring in interpolation
            during shape augmentation (scale, rotation etc.).

            """
            current_max_id = np.amax(ann)
            inst_list = list(np.unique(ann))
            inst_list.remove(0)  # 0 is background
            for inst_id in inst_list:
                inst_map = np.array(ann == inst_id, np.uint8)
                remapped_ids = ndimage.label(inst_map)[0]
                remapped_ids[remapped_ids > 1] += current_max_id
                ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
                current_max_id = np.amax(ann)
            return ann

        def get_bounding_box(img):
            """Get bounding box coordinate information."""
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # due to python indexing, need to add 1 to max
            # else accessing will be 1px in the box, not out
            rmax += 1
            cmax += 1
            return [rmin, rmax, cmin, cmax]

        orig_ann = ann.copy()  # instance ID map
        fixed_ann = fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map

        # TODO: deal with 1 label warning
        # crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

        x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(fixed_ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= ann.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= ann.shape[1] - 2:
                inst_box[3] += 2

            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(ndimage.center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.dstack([x_map, y_map])
        return hv_map

    def __call__(self, results):
        gt_instance = results["gt_instance"]
        hv_map = self.gen_instance_hv_map(gt_instance)
        results["gt_hv_map"] = hv_map
        results["seg_fields"].append("gt_hv_map")
        return results


@njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = (
            1
            / 9.0
            * (
                T[y * Lx + x]
                + T[(y - 1) * Lx + x]
                + T[(y + 1) * Lx + x]
                + T[y * Lx + x - 1]
                + T[y * Lx + x + 1]
                + T[(y - 1) * Lx + x - 1]
                + T[(y - 1) * Lx + x + 1]
                + T[(y + 1) * Lx + x - 1]
                + T[(y + 1) * Lx + x + 1]
            )
        )
    return T


@PIPELINES.register_module()
class GetCellPoseMap(object):
    def __init__(self):
        super(GetCellPoseMap, self).__init__()

    def _extend_centers_gpu(self, neighbors, centers, isneighbor, Ly, Lx, n_iter=200):
        """runs diffusion on GPU to generate flows for training images or quality control

        neighbors is 9 x pixels in masks,
        centers are mask centers,
        isneighbor is valid neighbor boolean 9 x pixels

        """
        nimg = neighbors.shape[0] // 9

        T = np.zeros((nimg, Ly, Lx), dtype=np.float32)
        meds = centers.astype(np.int64)
        for i in range(n_iter):
            T[:, meds[:, 0], meds[:, 1]] += 1
            Tneigh = T[:, neighbors[:, :, 0], neighbors[:, :, 1]]
            Tneigh *= isneighbor
            T[:, neighbors[0, :, 0], neighbors[0, :, 1]] = Tneigh.mean(axis=1)

        T = np.log(1.0 + T)
        # gradient positions
        grads = T[:, neighbors[[2, 1, 4, 3], :, 0], neighbors[[2, 1, 4, 3], :, 1]]

        dy = grads[:, 0] - grads[:, 1]
        dx = grads[:, 2] - grads[:, 3]
        del grads
        mu_torch = np.stack((dy.squeeze(), dx.squeeze()), axis=-2)
        return mu_torch

    def masks_to_flows(self, masks):
        Ly, Lx = masks.shape
        mu = np.zeros((2, Ly, Lx), np.float64)
        mu_c = np.zeros((Ly, Lx), np.float64)

        nmask = masks.max()
        slices = find_objects(masks)
        # dia = utils.diameters(masks)[0]
        # s2 = (.15 * dia)**2
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
                y, x = np.nonzero(masks[sr, sc] == (i + 1))
                y = y.astype(np.int32) + 1
                x = x.astype(np.int32) + 1
                ymed = np.median(y)
                xmed = np.median(x)
                imin = np.argmin((x - xmed) ** 2 + (y - ymed) ** 2)
                xmed = x[imin]
                ymed = y[imin]

                d2 = (x - xmed) ** 2 + (y - ymed) ** 2
                # mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

                niter = 2 * np.int32(np.ptp(x) + np.ptp(y))
                T = np.zeros((ly + 2) * (lx + 2), np.float64)
                T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(niter))
                T[(y + 1) * lx + x + 1] = np.log(1.0 + T[(y + 1) * lx + x + 1])

                dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
                dx = T[y * lx + x + 1] - T[y * lx + x - 1]
                mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))

        mu /= 1e-20 + (mu**2).sum(axis=0) ** 0.5

        return mu

    def __call__(self, results):
        results["gt_instance"] = fastremap.renumber(results["gt_instance"])[0]
        gt_instance = results["gt_instance"]
        vec = self.masks_to_flows(gt_instance).astype(np.float32)
        vec = np.moveaxis(vec, 0, -1)
        results["gt_vec"] = vec

        results["seg_fields"].append("gt_vec")

        return results


@PIPELINES.register_module()
class Normalize99(object):
    """normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    invert: invert image (useful if cells are dark instead of bright)

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """

    def __init__(self):
        self.mean = np.array([0, 0, 0], dtype=np.float32)
        self.std = np.array([0, 0, 0], dtype=np.float32)
        self.to_rgb = False

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # h, w, 3
        img = results["img"].astype(np.float32)
        for channel in range(img.shape[2]):
            img_c = img[..., channel]
            i99 = np.percentile(img_c, 99)
            i1 = np.percentile(img_c, 1)
            if i99 - i1 > 1e-3:
                norm_img_c = (img_c - i1) / (i99 - i1)
                img[..., channel] = norm_img_c
                self.mean[channel] = i1
                self.std[channel] = i99 - i1
            else:
                img[..., channel] = 0
                self.mean[channel] = (i1 + i99) / 2
                self.std[channel] = 1

        results["img"] = img

        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class GetHeatMap(object):
    def __init__(
        self,
        size,
        num_classes,
        grid_size,
        min_area=16,
        num_mask_per_img=25,
        num_neg_prompt=0,
        test_mode=False,
        iou_threshold=0.3,
        fix_mirror=True,
    ):
        super(GetHeatMap, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.min_area = min_area
        self.num_mask_per_img = num_mask_per_img
        self.num_neg_prompt = num_neg_prompt
        self.test_mode = test_mode  # 若为True，不计算热图的gt
        self.iou_threshold = iou_threshold
        self.fix_mirror = fix_mirror

    def __call__(self, results):
        def fix_mirror_padding(ann):
            """Deal with duplicated instances due to mirroring in interpolation
            during shape augmentation (scale, rotation etc.).

            """
            current_max_id = np.amax(ann)
            inst_list = list(np.unique(ann))
            inst_list.remove(0)  # 0 is background
            for inst_id in inst_list:
                inst_map = np.array(ann == inst_id, np.uint8)
                remapped_ids = ndimage.label(inst_map)[0]
                remapped_ids[remapped_ids > 1] += current_max_id
                ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
                current_max_id = np.amax(ann)
            return ann

        gt_instance = results["gt_instance"]
        h, w = gt_instance.shape[:2]
        if self.fix_mirror:
            gt_instance = fix_mirror_padding(gt_instance)
        if self.num_classes > 2:
            gt_ins_to_type = results.get(
                "gt_ins_to_type", [1] * (gt_instance.max() + 1)
            )
        else:
            gt_ins_to_type = [1] * (gt_instance.max() + 1)

        inst_types = []  # 每个instance的label, list of int
        inst_masks = []  # 每个instance的mask, list of numpy.array(256, 256)

        for i, inst_id in enumerate(np.unique(gt_instance)):
            if inst_id == 0:
                continue
            tmp_mask = gt_instance == inst_id
            if tmp_mask.sum() < self.min_area:
                continue
            inst_type = gt_ins_to_type[inst_id]
            inst_masks.append(((tmp_mask) * 1).astype(np.int32))
            inst_types.append(inst_type)

        if (self.test_mode is False) and (len(inst_masks) > 0):
            heat_map, inst_mask, is_center, center_xy, gt_bboxes = self.process_label(
                np.array(inst_types), np.array(inst_masks), self.iou_threshold
            )
        else:
            heat_map = np.zeros(
                [self.num_classes - 1, self.grid_size, self.grid_size], dtype=np.float32
            )
            inst_mask = np.zeros((0, self.size[0], self.size[1]), dtype=np.int16)
            is_center = np.zeros([self.grid_size**2], dtype=np.bool)
            center_xy = np.zeros((0, 2), dtype=np.float32)
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        # save_dir = Path("bishe")
        # save_dir.mkdir(exist_ok=True)
        # fig = plt.figure(figsize=(1.0, 1.0), dpi=256)
        # ax = fig.add_subplot(111)
        # ax.axis("off")
        # ax.imshow(heat_map[0], cmap="jet", interpolation="nearest")
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # plt.savefig(
        #     save_dir / f"{results['img_info']['filename'][:4]}_heat.png",
        #     dpi=256,
        #     bbox_inches="tight",
        #     pad_inches=0,
        # )
        # plt.close()

        # for prompter
        results["gt_heat_map"] = heat_map  # [num_classes, 64, 64]  热图
        results["gt_inst_mask"] = (
            inst_mask  # [num, 256, 256] 每个instance 的mask,num>=目标数
        )
        results["gt_is_center"] = is_center  # [4096, ]
        results["center_xy"] = center_xy  # [num, 2]
        results["cell_num"] = len(np.unique(gt_instance)) - 1
        results["gt_bboxes"] = gt_bboxes

        # for sam
        if results["cell_num"] > 0:
            all_points = []
            cell_types = []  # todo
            unique_pids = np.unique(gt_instance)[1:]  # remove zero
            for i, inst_id in enumerate(unique_pids):
                if inst_id == 0:
                    continue
                tmp_mask = gt_instance == inst_id
                if not self.test_mode:
                    pt = random.choice(np.argwhere(tmp_mask))[None, [1, 0]]  # [1, 2] x，y（从细胞中随机选取一个点）
                else:
                    (
                        center_w,
                        center_h,
                        width,
                        height,
                    ) = self.get_ins_info(tmp_mask, method="bbox")
                    pt = np.array([[center_w, center_h]])   # 如果是test，就选取细胞的中心点
                all_points.append(pt)
            all_points = np.concatenate(all_points, axis=0).astype(np.float32)
            if not self.test_mode:     # 如果是train的话，随机选取所有对象的提示点中的一部分
                chosen_pids = np.random.choice(
                    unique_pids,
                    min(results["cell_num"], self.num_mask_per_img),
                    replace=False,
                )
            else:
                chosen_pids = unique_pids

            inst_masks = []
            prompt_points = []
            prompt_boxes = []
            for pid in chosen_pids:
                # for point
                tmp_mask = np.equal(gt_instance, pid)
                inst_masks.append(tmp_mask)
                (
                    center_w,
                    center_h,
                    width,
                    height,
                ) = self.get_ins_info(tmp_mask, method="bbox")
                if not self.test_mode:
                    prompt_points.append(
                        random.choice(np.argwhere(tmp_mask))[None, [1, 0]].astype(
                            np.float32
                        )
                    )
                else:
                    prompt_points.append(
                        np.array([[center_w, center_h]]).astype(np.float32)
                    )

                # for box
                xyxy = [
                    center_w - width / 2,
                    center_h - height / 2,
                    center_w + width / 2,
                    center_h + height / 2,
                ]

                prompt_boxes.append(xyxy)

            inst_masks = np.stack(inst_masks, axis=0)  # [n, 256, 256]，n是提示点数目
            prompt_points = np.stack(prompt_points, axis=0)  # [n, 1, 2]
            prompt_labels = np.ones(prompt_points.shape[:2])  # [n, 1]，是SAM中表示物体是正还是负提示点的label
            prompt_boxes = np.stack(prompt_boxes, axis=0)  # [n, 4]
            prompt_boxes = box_noise(prompt_boxes, 0.1, a_min=0, a_max=h)  # no noise 测试上限
            if self.num_neg_prompt > 0 and np.random.rand() > 0.5:
                global_indices = [
                    np.where(unique_pids == pid)[0][0] for pid in chosen_pids
                ]

                prompt_points, prompt_labels = self.add_k_nearest_neg_prompt(
                    torch.from_numpy(prompt_points),
                    global_indices,
                    torch.from_numpy(all_points),
                    k=self.num_neg_prompt,
                )
            else:
                # padding
                prompt_points = np.concatenate(
                    [prompt_points, np.zeros_like(prompt_points)], 1
                )
                prompt_labels = np.concatenate(
                    [prompt_labels, -np.ones_like(prompt_labels)], 1
                )

        else:
            prompt_points = np.empty((0, (self.num_neg_prompt + 1), 2))
            prompt_labels = np.empty((0, (self.num_neg_prompt + 1)))
            prompt_boxes = np.empty((0, 4))
            all_points = np.empty((0, 2))
            inst_masks = np.empty((0, 256, 256))
            cell_types = np.empty(0)

        results["gt_sam_inst_masks"] = inst_masks
        results["gt_sam_prompt_points"] = prompt_points
        results["gt_sam_prompt_labels"] = prompt_labels
        results["gt_sam_prompt_bboxes"] = prompt_boxes

        return results

    def process_label(self, gt_labels_raw, gt_masks_raw, iou_threshold=0.3, tau=0.5):
        h, w = self.size

        heat_map = np.zeros(
            [self.num_classes - 1, self.grid_size, self.grid_size], dtype=np.float32
        )
        inst_mask = np.zeros(
            [self.grid_size**2, w, h], dtype=np.int16
        )  # 每个点对应一个mask
        is_center = np.zeros([self.grid_size**2], dtype=np.bool)  # 每个点的label

        center_xy = []
        xyxy = []

        if gt_masks_raw is not None:
            gt_labels = gt_labels_raw
            gt_masks = gt_masks_raw
            for seg_mask, gt_label in zip(gt_masks, gt_labels):
                (
                    center_w,
                    center_h,
                    width,
                    height,
                ) = self.get_ins_info(  # 这张mask的bbox的中心和宽高
                    seg_mask, method="bbox"
                )
                # if center_h <= 4 or center_w <= 4:
                #     continue

                # 框作为prompt
                center_xy.append([center_w, center_h])
                xyxy.append(
                    [
                        center_w - width / 2,
                        center_h - height / 2,
                        center_w + width / 2,
                        center_h + height / 2,
                    ]
                )

                # 热图相关
                radius = max(self.gaussian_radius((width, height), iou_threshold), 0)
                coord_h = int(
                    (center_h / h) / (1.0 / self.grid_size)
                )  # 对应64x64的高和宽
                coord_w = int((center_w / w) / (1.0 / self.grid_size))
                temp = self.draw_gaussian(
                    heat_map[gt_label - 1],
                    (coord_w, coord_h),
                    (radius / 4),  # 这里修改了heat_map
                )
                non_zeros = (temp > tau).nonzero()
                label = (
                    non_zeros[0] * self.grid_size
                    + non_zeros[1]  # 这个点在inst_mask的位置
                )  # label = int(coord_h * grid_size + coord_w)
                inst_mask[label, :, :] = seg_mask
                is_center[label] = True

            # 避免报错
            if is_center.sum() > 0:
                inst_mask = np.stack(
                    inst_mask[is_center], 0
                )  # [num, 256, 256] 每个instance 的mask
            else:
                inst_mask = np.zeros((0, h, w), dtype=np.int16)

            center_xy = np.array(center_xy)
            xyxy = np.array(xyxy)

        return (
            heat_map,
            inst_mask,
            is_center,
            center_xy,
            xyxy,
        )  # heat_map 热图[num_class, 64, 64]  is_center表示哪些点是instance

    def get_ins_info(self, seg_mask, method="bbox"):
        methods = ["bbox", "circle", "area"]
        assert (
            method in methods
        ), f"instance segmentation information should in {methods}"
        if method == "circle":
            contours, hierachy = cv2.findContours(
                (seg_mask * 255).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            (center_w, center_h), EC_radius = cv2.minEnclosingCircle(contours[0])
            return center_w, center_h, EC_radius * 2, EC_radius * 2
        elif method == "bbox":
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(
                np.array(seg_mask).astype(np.uint8)
            )
            center_w = bbox_x + bbox_w / 2
            center_h = bbox_y + bbox_h / 2
            return center_w, center_h, bbox_w, bbox_h
        elif method == "area":
            center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
            equal_diameter = (np.sum(seg_mask) / 3.1415) ** 0.5 * 2
            return center_w, center_h, equal_diameter, equal_diameter
        else:
            raise NotImplementedError

    def gaussian_radius(self, det_size, min_overlap=0.7):
        # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
        height, width = det_size

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    def draw_gaussian(self, heatmap, center, radius):
        def gaussian2D(shape, sigma=1.0):
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m : m + 1, -n : n + 1]
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        def insert_image(img, kernel, h, w):
            ks = kernel.shape[0]
            if ks != 0:
                half_ks = ks // 2
                img = np.pad(img, ((half_ks, half_ks), (half_ks, half_ks)))
                img[h : h + ks, w : w + ks] = kernel
                return img[half_ks:-half_ks, half_ks:-half_ks]
            else:
                img[h : h + 1, w : w + 1] = kernel
                return img

        diameter = float(2 * (int(radius) + 1) + 1)
        gaussian = gaussian2D(
            (diameter, diameter), sigma=radius / 3
        )  # gaussian_2d_kernel(int(diameter),radius/3)#
        coord_w, coord_h = center
        height, width = heatmap.shape
        temp = np.zeros((height, width), dtype=np.float32)
        temp = insert_image(temp, gaussian, coord_h, coord_w)
        np.maximum(heatmap, temp, out=heatmap)
        return temp

    @staticmethod
    def add_k_nearest_neg_prompt(
        prompt_points, global_indices, all_points, k: int = 1, min_dis=0.0
    ):
        if len(prompt_points) == 1:
            prompt_points = torch.cat(
                [prompt_points, torch.zeros((1, k, 2), device=prompt_points.device)],
                dim=1,
            )
            prompt_labels = torch.ones(
                prompt_points.shape[:2], dtype=torch.int32, device=prompt_points.device
            )
            prompt_labels[0, 1] = -1
        else:
            all_points = all_points.view(-1, 2)
            dis = torch.cdist(all_points, all_points, p=2.0)
            dis = dis.fill_diagonal_(np.inf)
            dis[dis < min_dis] = np.inf

            available_num = min(k, len(prompt_points) - 1)
            neg_prompt_points = all_points[
                torch.topk(
                    dis[global_indices], available_num, dim=1, largest=False
                ).indices,
                :,
            ]
            prompt_points = torch.cat(
                [
                    prompt_points,
                    neg_prompt_points,
                    torch.zeros(
                        (len(prompt_points), k - available_num, 2),
                        device=prompt_points.device,
                    ),
                ],
                dim=1,
            )

            prompt_labels = torch.ones(
                prompt_points.shape[:2], dtype=torch.int32, device=prompt_points.device
            )
            prompt_labels[:, 1 : available_num + 1] = 0
            prompt_labels[:, available_num + 1 :] = -1

        return prompt_points, prompt_labels


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation. Adds custom transformations from
    Albumentations library. Please, visit
    `https://albumentations.readthedocs.io` to get more information. An example
    of ``transforms`` is as followed:

    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
        update_pad_shape (bool): Whether to update padding shape according to \
            the output shape of the last transform
    """

    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        if Compose is None:
            raise ImportError(
                "albumentations is not installed, "
                "we suggest install albumentation by "
                '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
            )

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])
        # self.aug = Compose(
        #     [self.albu_builder(t) for t in self.transforms],
        #     additional_targets={
        #         "gt_instance": "mask",
        #         "gt_semantic_seg": "mask",
        #     }
        # )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
                "gt_masks": "masks",
            }
        else:
            self.keymap_to_albu = copy.deepcopy(keymap)
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise ImportError(
                    "albumentations is not installed, "
                    "we suggest install albumentation by "
                    '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
                )
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, _ in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # Convert to RGB since Albumentations works with RGB images
        results["image"] = cv2.cvtColor(results["image"], cv2.COLOR_BGR2RGB)

        results = self.aug(**results)

        # Convert back to BGR
        results["image"] = cv2.cvtColor(results["image"], cv2.COLOR_RGB2BGR)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class GetSemanticSeg():
    def __init__(self):
        super(GetSemanticSeg, self).__init__()

    def __call__(self, results):
        gt_instance = results['gt_instance']
        gt_ins_to_type = results.get("gt_ins_to_type", None)
        gt_semantic_seg = np.zeros_like(gt_instance).astype(np.uint8)
        if gt_ins_to_type is not None:
            for i in np.unique(gt_instance):

                if i in gt_ins_to_type.keys():
                    gt_semantic_seg = np.where(
                        gt_instance == i, gt_ins_to_type[i], gt_semantic_seg
                    )
                else:
                    continue
                    print(i)
        else:
            mask = gt_instance > 0
            gt_semantic_seg[mask] = 1

        results["gt_semantic_seg"] = gt_semantic_seg
        return results


@PIPELINES.register_module()
class GetCenter(object):
    def __init__(self):
        super(GetCenter, self).__init__()

    def __call__(self, results):
        def fix_mirror_padding(ann):
            """Deal with duplicated instances due to mirroring in interpolation
            during shape augmentation (scale, rotation etc.).

            """
            current_max_id = np.amax(ann)
            inst_list = list(np.unique(ann))
            inst_list.remove(0)  # 0 is background
            for inst_id in inst_list:
                inst_map = np.array(ann == inst_id, np.uint8)
                remapped_ids = ndimage.label(inst_map)[0]
                remapped_ids[remapped_ids > 1] += current_max_id
                ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
                current_max_id = np.amax(ann)
            return ann

        assert "gt_inst_mask" in results, "Please call GetHeatMap first"
        gt_inst_mask = results["gt_inst_mask"]

        gt_points = []
        for inst_mask_single in gt_inst_mask:
            center_w, center_h, equal_diameter, equal_diameter = self.get_ins_info(
                inst_mask_single, method="area"
            )
            gt_points.append([center_w.astype(np.float32), center_h.astype(np.float32)])

        results["gt_points"] = gt_points  # [num, 2]
        return results

    def get_ins_info(self, seg_mask, method="bbox"):
        methods = ["bbox", "circle", "area"]
        assert (
            method in methods
        ), f"instance segmentation information should in {methods}"
        if method == "circle":
            contours, hierachy = cv2.findContours(
                (seg_mask * 255).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            (center_w, center_h), EC_radius = cv2.minEnclosingCircle(contours[0])
            return center_w, center_h, EC_radius * 2, EC_radius * 2
        elif method == "bbox":
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(
                np.array(seg_mask).astype(np.uint8)
            )
            center_w = bbox_x + bbox_w / 2
            center_h = bbox_y + bbox_h / 2
            return center_w, center_h, bbox_w, bbox_h
        elif method == "area":
            center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
            equal_diameter = (np.sum(seg_mask) / 3.1415) ** 0.5 * 2
            return center_w, center_h, equal_diameter, equal_diameter
        else:
            raise NotImplementedError
