# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import json
import colorsys
import random
import multiprocess as mp
import math

import torch
from torch.utils.data import Dataset

from . import CityscapesDataset
from tqdm import tqdm

import os
import numpy as np
import cv2
import os.path as osp
from collections import OrderedDict
from functools import reduce
import warnings
import scipy
from scipy import ndimage
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter1d
import fastremap
from scipy import io as sio
from scipy.ndimage import center_of_mass

import mmcv
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
import torch.nn.functional as F

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from numba import njit
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations, LoadAnnotationsNpy
from .nuclei import (
    get_dice_1,
    get_dice_2,
    get_fast_dice_2,
    get_fast_pq,
    get_fast_aji,
    get_fast_aji_plus,
    remap_label,
    pair_coordinates,
)
from .custom import CustomDataset


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

    slices = ndimage.find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = ndimage.binary_fill_holes(msk[k])
                else:
                    msk = ndimage.binary_fill_holes(msk)
                masks[slc][msk] = j + 1
                j += 1
    return masks


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    random.seed(42)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]  # 最大值
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


@DATASETS.register_module()
class NucleiHV2Dataset(CustomDataset):
    # CLASSES = ("bg", "1", "2", "3", "4")
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, 
                classes=("epithelial", "lymphocyte", "neutrophil", "macrophage", "ambiguous"),  # monusac,
                type_colors=None,
                **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")

        self.eval_type = kwargs.get("eval_type", False)
        if "eval_type" in kwargs:
            kwargs.pop("eval_type")
        
        self.classes = kwargs.pop("classes", None)
        self.type_color = kwargs.pop("type_color", None)
        super(NucleiHV2Dataset, self).__init__(**kwargs)
        # self.num_classes = len(self.CLASSES)
        if self.classes:
            self.num_classes = len(self.classes)
        else:
            self.num_classes = 2

    def get_gt(self):
        """Get ground truth for evaluation."""
        gts = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
            if seg_map.endswith(".mat"):
                results = sio.loadmat(seg_map)
                gt = {
                    "inst_map": results["inst_map"],
                    "type_map": results.get("type_map"),
                    "inst_type": results.get("inst_type"),
                    "inst_centroid": results.get("inst_centroid"),
                }
                # 合并gt中的一些分类
                type_map = gt["type_map"]
                if self.num_classes == 4:
                    type_map = np.where((type_map == 3) | (type_map == 4), 3, type_map)
                    type_map = np.where((type_map == 5) | (type_map == 6) | (type_map == 7), 4, type_map)
                    gt["type_map"] = type_map

                    inst_label = gt["inst_type"]
                    inst_label = np.where(inst_label == 4, 3, inst_label)  # 把 4 变成 3
                    inst_label = np.where(inst_label >= 5, 4, inst_label)  # 把 >=5 变成 4
                    gt["inst_type"] = inst_label
                            
            elif seg_map.endswith(".npy"):
                inst_map = np.load(seg_map)
                gt = {"inst_map": inst_map}
            elif seg_map.endswith(".pth"):
                results = torch.load(seg_map)
                gt = {
                    "inst_map": results["inst_map"],
                    "type_map": results.get("type_map"),
                    "tissue_type": results.get("tissue_type"),
                    "inst_type": results.get("inst_type"),
                    "inst_centroid": results.get("inst_centroid"),
                }
            else:
                raise Exception(f"unknown path {seg_map}")
            gts.append(gt)
        return gts

    def evaluate(self, results, metric="mIoU", logger=None, gts=None, **kwargs):
        """计算细胞核分割的指标，包括dice, aji, dq, sq, pq, aji_plus.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): 4通道，前两个通道是语义分割结果(softmax结果)，后两通道是hv结果
            metric (str | list[str]):
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        eval_results = {}

        if gts is None:
            gts = self.get_gt()

        ret_metrics = self.run_nuclei_stat(results, gts)

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 3)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])

        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            eval_results[key] = value / 100.0

        return eval_results

    def run_nuclei_stat(self, results, gts):
        """
        计算各种指标，包括dice, aji, dq, sq, pq, aji_plus, 和分类的F1
        Args:
            results: results (list[numpy.Array]): 前两个通道是hv，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景)
            gts: list of dict("inst_map": ,
                             "type_map" ,
                            "inst_type",
                            "inst_centroid")
            metric:

        Returns:

        """
        paired_all = []  # unique matched index pair
        unpaired_true_all = (
            []
        )  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = (
            []
        )  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point
        metrics = [[], [], [], [], [], []]
        for file_idx, (res, gt) in enumerate(zip(results, gts)):
            pred = self.result_to_inst(res)

            # to ensure that the instance numbering is contiguous
            # pred = remap_label(pred, by_size=False)
            pred, _ = fastremap.renumber(pred)
            binary_map = pred.copy()
            binary_map[binary_map > 0] = 1
            label_idx = np.unique(pred)  # 0,1,2,...N

            # [(y,x), ...]
            inst_centroid_yx = center_of_mass(
                binary_map, pred, label_idx[1:]
            )
            inst_centroid_xy = [(each[1], each[0]) for each in inst_centroid_yx]

            # 分类的指标
            if self.eval_type:
                # dont squeeze, may be 1 instance exist
                true_centroid = (gt["inst_centroid"]).astype("float32")
                true_inst_type = (gt["inst_type"]).astype("int32")
                if true_centroid.shape[0] != 0:
                    true_inst_type = true_inst_type[:, 0]
                else:  # no instance at all
                    true_centroid = np.array([[0, 0]])
                    true_inst_type = np.array([0])

                # dont squeeze, may be 1 instance exist
                pred_centroid = np.array(inst_centroid_xy).astype("float32")
                pred_inst_type = self.get_inst_type(res, pred)
                if pred_centroid.shape[0] == 0:
                    pred_centroid = np.array([[0, 0]])
                    pred_inst_type = np.array([0])

                # ! if take longer than 1min for 1000 vs 1000 pairing, sth is wrong with coord
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_centroid, pred_centroid, 12
                )

                # * Aggreate information
                # get the offset as each index represent 1 independent instance
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all[-1].shape[0]
                    if file_idx != 0
                    else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all[-1].shape[0]
                    if file_idx != 0
                    else 0
                )
                true_inst_type_all.append(true_inst_type)
                pred_inst_type_all.append(pred_inst_type)

                # increment the pairing index statistic
                if paired.shape[0] != 0:  # ! sanity
                    paired[:, 0] += true_idx_offset
                    paired[:, 1] += pred_idx_offset
                    paired_all.append(paired)

                unpaired_true += true_idx_offset
                unpaired_pred += pred_idx_offset
                unpaired_true_all.append(unpaired_true)
                unpaired_pred_all.append(unpaired_pred)

            # gt_ins_map = remap_label(gt["inst_map"], by_size=False)
            gt_ins_map, _ = fastremap.renumber(gt["inst_map"])
            if (gt_ins_map.max() == 0) or (pred.max() == 0):
                continue

            # 分割的指标
            metrics[0].append(get_dice_1(gt_ins_map, pred))
            metrics[1].append(get_fast_aji(gt_ins_map, pred))

            pq_info = get_fast_pq(gt_ins_map, pred, match_iou=0.5)[0]
            metrics[2].append(pq_info[0])  # dq
            metrics[3].append(pq_info[1])  # sq
            metrics[4].append(pq_info[2])  # pq
            metrics[5].append(get_fast_aji_plus(gt_ins_map, pred))

            # print(self.img_infos[file_idx]['ann']['seg_map'], end="\t")
            # for scores in metrics:
            #     print("%f " % scores[-1], end="  ")
            # print()

        ####
        metrics = np.array(metrics)
        metrics_avg = np.nanmean(metrics, axis=-1)
        ret_metrics = {
            "DICE": metrics_avg[0],
            "AJI": metrics_avg[1],
            "DQ": metrics_avg[2],
            "SQ": metrics_avg[3],
            "PQ": metrics_avg[4],
        }
        # 分类的指标
        if len(paired_all) > 0:
            paired_all = np.concatenate(paired_all, axis=0)
            unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
            unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
            true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
            pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

            paired_true_type = true_inst_type_all[paired_all[:, 0]]
            paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
            unpaired_true_type = true_inst_type_all[unpaired_true_all]
            unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

            # overall
            # * quite meaningless for not exhaustive annotated dataset
            w = [1, 1]
            tp_d = paired_pred_type.shape[0]
            fp_d = unpaired_pred_type.shape[0]
            fn_d = unpaired_true_type.shape[0]

            tp_tn_dt = (paired_pred_type == paired_true_type).sum()
            fp_fn_dt = (paired_pred_type != paired_true_type).sum()

            acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
            f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

            w = [2, 2, 1, 1]

            type_uid_list = np.unique(true_inst_type_all)
            if np.any(type_uid_list == 0):
                type_uid_list = type_uid_list[1:]  # 去掉0

            # 分类指标
            results_list = [
                f1_d,
            ]
            for type_uid in type_uid_list:
                f1_type = self._f1_type(
                    paired_true_type,
                    paired_pred_type,
                    unpaired_true_type,
                    unpaired_pred_type,
                    type_uid,
                    w,
                )
                results_list.append(f1_type)

            ret_metrics = {
                "DICE": metrics_avg[0],
                "AJI": metrics_avg[1],
                "DQ": metrics_avg[2],
                "SQ": metrics_avg[3],
                "PQ": metrics_avg[4],
                "Detection": f1_d,
            }
            for i in range(1, len(results_list)):
                ret_metrics[f"Fc{i}"] = results_list[i]

        return ret_metrics

    def result_to_inst(self, result):
        def remove_small_objects(pred, min_size=16, connectivity=1):
            """Remove connected components smaller than the specified size.

            This function is taken from skimage.morphology.remove_small_objects, but the warning
            is removed when a single label is provided.

            Args:
                pred: input labelled array
                min_size: minimum size of instance in output array
                connectivity: The connectivity defining the neighborhood of a pixel.

            Returns:
                out: output array with instances removed under min_size

            """
            out = pred

            if min_size == 0:  # shortcut for efficiency
                return out

            if out.dtype == bool:
                selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
                ccs = np.zeros_like(pred, dtype=np.int32)
                ndimage.label(pred, selem, output=ccs)
            else:
                ccs = out

            try:
                component_sizes = np.bincount(ccs.ravel())
            except ValueError:
                raise ValueError(
                    "Negative value labels are not supported. Try "
                    "relabeling the input with `scipy.ndimage.label` or "
                    "`skimage.morphology.label`."
                )

            too_small = component_sizes < min_size
            too_small_mask = too_small[ccs]
            out[too_small_mask] = 0

            return out

        def __proc_np_hv(pred):
            """Process Nuclei Prediction with XY Coordinate Map.

            Args:
                pred: prediction output, assuming
                      channel 0 contain probability map of nuclei
                      channel 1 containing the regressed X-map
                      channel 2 containing the regressed Y-map

            """
            pred = np.array(pred, dtype=np.float32)

            blb_raw = pred[..., 0]
            h_dir_raw = pred[..., 1]
            v_dir_raw = pred[..., 2]

            # processing
            blb = np.array(blb_raw >= 0.5, dtype=np.int32)

            blb = ndimage.label(blb)[0]
            blb = remove_small_objects(blb, min_size=10)
            blb[blb > 0] = 1  # background is 0 already

            h_dir = cv2.normalize(
                h_dir_raw,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            v_dir = cv2.normalize(
                v_dir_raw,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

            sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
            sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

            sobelh = 1 - (
                cv2.normalize(
                    sobelh,
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )
            )
            sobelv = 1 - (
                cv2.normalize(
                    sobelv,
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )
            )

            overall = np.maximum(sobelh, sobelv)
            overall = overall - (1 - blb)
            overall[overall < 0] = 0

            dist = (1.0 - overall) * blb
            ## nuclei values form mountains so inverse to get basins
            dist = -cv2.GaussianBlur(dist, (3, 3), 0)

            overall = np.array(overall >= 0.4, dtype=np.int32)

            marker = blb - overall
            marker[marker < 0] = 0
            marker = ndimage.binary_fill_holes(marker).astype("uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
            marker = ndimage.label(marker)[0]
            marker = remove_small_objects(marker, min_size=10)

            proced_pred = watershed(dist, markers=marker, mask=blb)

            return proced_pred

        # result前两个通道是hv，第三个通道是前景概率(未经sigmoid)
        pred_inst = np.zeros_like(result[:3])
        pred_inst[0, ...] = torch.sigmoid(torch.from_numpy(result[2, ...])).numpy()
        pred_inst[1:, ...] = result[:2, ...]
        pred_inst = np.transpose(pred_inst, (1, 2, 0))
        pred_inst = np.squeeze(pred_inst)
        pred_inst = __proc_np_hv(pred_inst)

        # print('here')
        # ! WARNING: ID MAY NOT BE CONTIGUOUS
        # inst_id in the dict maps to the same value in the `pred_inst`
        return pred_inst.astype("int32")

    def get_inst_type(self, result, pred):
        """
        计算每个nuclei instance 的类别

        Args:
            result: (numpy.Array): 前两个通道是梯度，然后两个通道是前景背景，后N个通道是常规等的语义分割(包括背景)
            pred:  (numpy.Array): nuclei instance background: 0, nuclei: 1-N

        Returns: (numpy.Array [N,]) types of each nuclei

        """
        pred_type = torch.from_numpy(result[3:])
        pred_type = torch.argmax(pred_type, dim=0).numpy()  # H, W
        inst_id_list = np.unique(pred)[1:]  # exclude bg
        slices = ndimage.find_objects(pred)

        pred_inst_type = []  # 函数返回值，List[N]，每个instance的类型
        for i, slc in enumerate(slices):
            inst_map = pred[slc]
            inst_type = pred_type[slc]
            inst_type = inst_type[inst_map == (i + 1)]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))  # [(type, num_pixel),...]
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_inst_type.append(inst_type)
        return np.array(pred_inst_type, dtype=np.int32)

    def _f1_type(
            self, paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w
    ):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type

    def format_results(self, results, **kwargs):
        for file_idx, res in enumerate(results):
            pred = self.result_to_inst(res)
            # pred = remap_label(pred, by_size=False)
            pred, _ = fastremap.renumber(pred)
            binary_map = pred.copy()
            binary_map[binary_map > 0] = 1
            label_idx = np.unique(pred)  # 0,1,2,...N

            # 根据mapping调整res键的编号
            for key, value in res.items():
                if key != 'inst_preds':
                    modified = {mapping.get(k, k): v for k, v in value.items()}
                    res[key] = modified
                    
            # [(y,x), ...]
            inst_centroid_yx = center_of_mass(
                binary_map, pred, label_idx[1:]
            )
            inst_centroid_xy = [(each[1], each[0]) for each in inst_centroid_yx]

            os.makedirs(kwargs["imgfile_prefix"], exist_ok=True)
            img_info = self.img_infos[file_idx]
            save_path = os.path.join(
                kwargs["imgfile_prefix"], img_info["filename"].split(".")[0] + ".npy"
            )
            np.save(save_path, pred)
            pred_json = {"nuc": dict()}

            inst_id_list = np.unique(pred)[1:]  # exclude background

            if len(inst_id_list) > 10000:
                num_processes = 16
            else:
                num_processes = 1

            processes = []
            queue = mp.Queue()
            split_size = math.ceil(len(inst_id_list) / num_processes)
            for k in range(num_processes):
                process = mp.Process(
                    target=self._run_task,
                    args=(
                        res,
                        pred,
                        k,
                        inst_id_list[k * split_size:(k + 1) * split_size],
                        queue
                    ),
                    daemon=True,
                )
                processes.append(process)

            for process in processes:
                process.start()

            # for process in processes:
            #     process.join()

            inst_info_dict = {}
            for k in range(num_processes):
                inst_info_dict.update(queue.get())

            self.__save_json(
                os.path.join(
                    kwargs["imgfile_prefix"],
                    img_info["filename"].split(".")[0] + ".json",
                ),
                inst_info_dict,
            )
            img = cv2.imread(os.path.join(self.img_dir, img_info["filename"]))
            overlay = self.visualize_instances_dict(img, inst_info_dict, False, self.type_color)
            cv2.imwrite(
                os.path.join(kwargs["imgfile_prefix"], img_info["filename"]), overlay
            )

    def _run_task(self, res, pred, index, inst_id_list, queue):
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = self.get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                       inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
                       ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[int(inst_id)] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": res['inst_softmax'].get(inst_id, []),
                "type": res['inst_labels'].get(inst_id, -1),
            }

        queue.put(inst_info_dict)

    def get_bounding_box(self, img):
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

    def __save_json(self, path, old_dict, mag=None):
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = {"mag": mag, "nuc": new_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict

    def visualize_instances_dict(
            self, input_image, inst_dict, draw_dot=True, type_colour=None, line_thickness=2
    ):
        """Overlays segmentation results (dictionary) on image as contours.

        Args:
            input_image: input image
            inst_dict: dict of output prediction, defined as in this library
            draw_dot: to draw a dot for each centroid
            type_colour: a dict of {type_id : (type_name, colour)} ,
                         `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
            line_thickness: line thickness of contours
        """
        overlay = np.copy((input_image))
        # overlay = np.zeros(input_image.shape, dtype=np.uint8)
        inst_rng_colors = random_colors(len(inst_dict))
        inst_rng_colors = np.array(inst_rng_colors) * 255
        inst_rng_colors = inst_rng_colors.astype(np.uint8)

        for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
            inst_contour = inst_info["contour"]
            if "type" in inst_info and type_colour is not None:
                inst_colour = type_colour[str(inst_info["type"]+1)][1]
            else:
                inst_colour = (inst_rng_colors[idx]).tolist()

            # inst_colour = (0, 0, 0)  # 黑色
            cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

            if draw_dot:
                inst_centroid = inst_info["centroid"]
                inst_centroid = tuple([int(v) for v in inst_centroid])
                overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
        return overlay
