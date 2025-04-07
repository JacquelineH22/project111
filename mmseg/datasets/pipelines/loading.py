# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import mmcv
import numpy as np
from scipy import ndimage
from scipy import io as sio
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        file_client_args=dict(backend="disk"),
        imdecode_backend="cv2",
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend
        )
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32},"
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(
        self,
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = (
            mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
            .squeeze()
            .astype(np.uint8)
        )
        # modify if custom classes
        if results.get("label_map", None) is not None:
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label},"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotationsNpy(object):
    """Load annotations for semantic segmentation.
    Npy数据仅训练二分类的语义分割！

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(
        self,
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]
        # img_bytes = self.file_client.get(filename)
        # gt_semantic_seg = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).squeeze().astype(np.uint8)

        gt_instance = np.load(filename).astype(np.int32)
        gt_semantic_seg = np.zeros_like(gt_instance, dtype=np.uint8)
        mask = gt_instance > 0
        gt_semantic_seg[mask] = 1

        # modify if custom classes
        # if results.get('label_map', None) is not None:
        #     for old_id, new_id in results['label_map'].items():
        #         gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # # reduce zero_label
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255

        results["gt_instance"] = gt_instance
        results["gt_semantic_seg"] = gt_semantic_seg
        # hv_map 在后面加，这样不影响数据增强
        # results['gt_hv_map'] = hv_map
        results["seg_fields"].append("gt_instance")
        results["seg_fields"].append("gt_semantic_seg")
        # results['seg_fields'].append('gt_hv_map')

        # _, ax = plt.subplots(1, 3, figsize=(11, 11))
        # ax[0].imshow(gt_instance)
        # ax[1].imshow(hv_map[..., 0])
        # ax[2].imshow(hv_map[..., 1])
        # plt.show()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label},"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

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
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
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


@PIPELINES.register_module()
class LoadAnnotationsMat(object):
    """
    TODO: 读入多分类GT semantic seg
    """
    def __init__(
        self,
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
    
    def inst_to_type(self, inst_map, type_map):
        inst_to_type = {}
        for inst_id in np.unique(inst_map):
            if inst_id == 0: 
                continue
            mask = (inst_map == inst_id)
            # 找到该实例区域内最多出现的类型标签
            type_label = np.bincount(type_map[mask].astype(int)).argmax()
            inst_to_type[inst_id] = type_label
        
        return inst_to_type

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        gt = sio.loadmat(filename)
        gt_instance = gt["inst_map"].astype(np.int32)
        gt_ins_to_type = gt.get("gt_ins_to_type", None)
        gt_semantic_seg = gt.get("type_map", None)
        # 针对MoNuSAC
        gt_semantic_seg[(gt_semantic_seg == 3) | (gt_semantic_seg == 4)] = 3
        gt_semantic_seg[(gt_semantic_seg == 5) | (gt_semantic_seg == 6) | (gt_semantic_seg == 7)] = 4
        gt_ins_to_type = self.inst_to_type(gt_instance, gt_semantic_seg)

        gt_ins_centroid = gt.get("inst_centroid", None)

        if gt_semantic_seg is None:
            gt_semantic_seg = np.zeros_like(gt_instance, dtype=np.int32)
            mask = gt_instance > 0
            gt_semantic_seg[mask] = 1
        if gt_ins_to_type is None:
            gt_ins_to_type = self.inst_to_type(gt_instance, gt_semantic_seg)

        results["gt_instance"] = gt_instance
        results["gt_semantic_seg"] = gt_semantic_seg.astype(np.int32)
        results["gt_ins_to_type"] = gt_ins_to_type
        results["gt_ins_centroid"] = gt_ins_centroid

        results["seg_fields"].append("gt_instance")
        results["seg_fields"].append("gt_semantic_seg")

        return results


@PIPELINES.register_module()
class LoadAnnotationsPth(object):
    def __init__(
        self,
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
    
    def inst_to_type(self, inst_map, type_map):
        inst_to_type = {}
        for inst_id in np.unique(inst_map):
            if inst_id == 0: 
                continue
            mask = (inst_map == inst_id)
            # 找到该实例区域内最多出现的类型标签
            type_label = np.bincount(type_map[mask]).argmax()
            inst_to_type[inst_id] = type_label
        
        return inst_to_type

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        gt = torch.load(filename)
        gt_instance = gt["inst_map"].astype(np.int32)
        gt_semantic_seg = gt.get("type_map", None)
        gt_ins_to_type = gt.get("gt_ins_to_type", None)
        tissue_type = gt.get("tissue_type", None)

        if gt_semantic_seg is None:
            gt_semantic_seg = np.zeros_like(gt_instance, dtype=np.int32)
            mask = gt_instance > 0
            gt_semantic_seg[mask] = 1

        if gt_ins_to_type is None:
            gt_ins_to_type = self.inst_to_type(gt_instance, gt_semantic_seg)
        
        results["gt_instance"] = gt_instance
        results["gt_semantic_seg"] = gt_semantic_seg
        results["gt_ins_to_type"] = gt_ins_to_type
        results["tissue_type"] = tissue_type

        results["seg_fields"].append("gt_instance")
        results["seg_fields"].append("gt_semantic_seg")

        return results
