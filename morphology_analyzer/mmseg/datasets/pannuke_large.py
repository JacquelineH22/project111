import json
import colorsys
import random
from multiprocessing import Pool
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import Dataset

from . import CityscapesDataset
from tqdm import tqdm

import os
import numpy as np
import cv2
import os.path as osp
from collections import OrderedDict
from scipy.ndimage import measurements
from fastremap import renumber
from mmcv.utils import print_log
from prettytable import PrettyTable

import pandas as pd

from .builder import DATASETS
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


def get_inst_info(pred, inst_id, res):
    inst_map = pred == inst_id
    rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
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
        return None, None
    if len(inst_contour.shape) != 2:
        return None, None
    inst_centroid = [
        (inst_moment["m10"] / inst_moment["m00"]),
        (inst_moment["m01"] / inst_moment["m00"]),
    ]
    inst_centroid = np.array(inst_centroid)
    inst_contour[:, 0] += inst_bbox[0][1]  # X
    inst_contour[:, 1] += inst_bbox[0][0]  # Y
    inst_centroid[0] += inst_bbox[0][1]  # X
    inst_centroid[1] += inst_bbox[0][0]  # Y
    inst_info = {
        "bbox": inst_bbox,
        "centroid": inst_centroid,
        "contour": inst_contour,
        "type_prob": res['inst_softmax'][inst_id],
        "type": res['inst_labels'][inst_id]}
        # "type_prob": None,
        # "type": None}
    return inst_id, inst_info


@DATASETS.register_module()
class PanNukeLargeDataset(CustomDataset):
    # CLASSES = ("bg", "1", "2", "3", "4")
    PALETTE = CityscapesDataset.PALETTE

    # wget https://vscode.cdn.azure.cn/stable/863d2581ecda6849923a2118d93a088b0745d9d6/vscode-server-linux-x64.tar.gz
    def __init__(
            self,
            classes=("Neo", "Inflam", "Connec", "Necros", "Epi"),    # pannuke
            **kwargs,
    ):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(PanNukeLargeDataset, self).__init__(classes=classes, **kwargs)
        self.tissue_types = {
            "Adrenal_gland": 0,
            "Bile-duct": 1,
            "Bladder": 2,
            "Breast": 3,
            "Cervix": 4,
            "Colon": 5,
            "Esophagus": 6,
            "HeadNeck": 7,
            "Kidney": 8,
            "Liver": 9,
            "Lung": 10,
            "Ovarian": 11,
            "Pancreatic": 12,
            "Prostate": 13,
            "Skin": 14,
            "Stomach": 15,
            "Testis": 16,
            "Thyroid": 17,
            "Uterus": 18,
        }
        self.num_classes = len(self.CLASSES)
        # self. img_infos = self.img_infos[:1000]
        self.type_colors = {
            "0" : ["nolabe", [0  ,   0,   0]], 
            "1" : ["neopla", [255,   0,   0]], 
            "2" : ["inflam", [0  , 255,   0]], 
            "3" : ["connec", [0  ,   0, 255]], 
            "4" : ["necros", [255, 255,   0]], 
            "5" : ["Epi", [255, 165,   0]] 
        }

    def get_gt(self):
        """Get ground truth for evaluation."""
        gts = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
            if seg_map.endswith(".pth"):
                results = torch.load(seg_map)
                gt = {
                    "inst_map": results["inst_map"],
                    "type_map": results.get("type_map"),
                    "tissue_type": results.get("tissue_type"),
                    "inst_type": results.get("inst_type"),
                    "inst_centroid": results.get("inst_centroid"),
                }
            elif seg_map.endswith(".npy"):
                inst_map = np.load(seg_map)
                gt = {"inst_map": inst_map}
            else:
                raise Exception(f"unknown path {seg_map}")
            gts.append(gt)
        return gts

    def evaluate(self, results, metric="mIoU", logger=None, gts=None, **kwargs):
        """计算细胞核分割的指标，包括各组织的bpq和mpq.

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
        计算各种指标，包括各组织的bpq和mpq.
        Args:
            results: dict
            gts: list of dict("inst_map": ,
                             "type_map" ,
                            "inst_type",
                            "inst_centroid")
            metric:

        Returns:

        """
        nuclei_pq_scores = []  # [2722, 5]
        # [19, n, 5] 19类组织，每类组织n张图(不同组织n不相等)，每张图5类的pq
        tissue_nuclei_pq_scores = [[] for _ in self.tissue_types]

        binary_pq_scores = []  # [2722,] image_id
        tissue_binary_pq_scores = [[] for _ in self.tissue_types]  # [19, n]
        for file_idx, (res, gt) in enumerate(zip(results, gts)):
            pred_inst_map = res["inst_preds"].astype("int32")
            pred_type = res["inst_labels"]  # dict， type从0开始

            # 每类的inst_map
            pred_inst_map_c = np.zeros((self.num_classes, *pred_inst_map.shape))
            for inst_id in np.unique(pred_inst_map):
                if inst_id == 0:
                    continue
                else:
                    ty = pred_type[inst_id]
                    pred_inst_map_c[ty, pred_inst_map == inst_id] = inst_id

            gt_inst_map = remap_label(gt["inst_map"], by_size=False)  # [256, 256]
            gt_type_map = gt["type_map"]  # [256, 256]
            gt_tissue_type = gt["tissue_type"]  # str

            # one-hot [256, 256, 6]
            gt_type_map = np.eye(self.num_classes + 1)[gt_type_map]
            gt_instance_types_nuclei = gt_type_map * np.expand_dims(gt_inst_map, -1)
            # [6, 256, 256]
            gt_instance_types_nuclei = gt_instance_types_nuclei.transpose(2, 0, 1)

            # 不考虑类别的指标
            if len(np.unique(pred_inst_map)) == 1:
                bpq_tmp = np.nan
                bdq_tmp = np.nan
                bsq_tmp = np.nan
            else:
                [bdq_tmp, bsq_tmp, bpq_tmp], _ = get_fast_pq(
                    gt_inst_map, remap_label(pred_inst_map)
                )

            binary_pq_scores.append(bpq_tmp)
            tissue_binary_pq_scores[self.tissue_types[gt_tissue_type]].append(bpq_tmp)

            # 考虑类别的指标
            nuclei_type_pq = []  # 一张图5类分别的pq
            nuclei_type_dq = []
            nuclei_type_sq = []
            for c in range(self.num_classes):
                pred_nuclei_instance_class = remap_label(pred_inst_map_c[c])
                target_nuclei_instance_class = remap_label(
                    gt_instance_types_nuclei[c + 1]
                )

                if len(np.unique(target_nuclei_instance_class)) == 1:
                    mpq_tmp = np.nan
                    mdq_tmp = np.nan
                    msq_tmp = np.nan
                else:
                    [mdq_tmp, msq_tmp, mpq_tmp], _ = get_fast_pq(
                        target_nuclei_instance_class, pred_nuclei_instance_class
                    )
                nuclei_type_pq.append(mpq_tmp)
                nuclei_type_dq.append(mdq_tmp)
                nuclei_type_sq.append(msq_tmp)

            nuclei_pq_scores.append(nuclei_type_pq)
            tissue_nuclei_pq_scores[self.tissue_types[gt_tissue_type]].append(
                nuclei_type_pq
            )

        tissue_mpq_scores = [0.0 for _ in self.tissue_types]  # [19, ]
        for tissue_type, tid in self.tissue_types.items():
            tmp = tissue_nuclei_pq_scores[tid]  # [num, 5]
            tissue_mpq_scores[tid] = np.nanmean(np.nanmean(tmp, axis=1))

        tissue_mPQ = np.nanmean(tissue_mpq_scores)

        tissue_bpq_scores = [0.0 for _ in self.tissue_types]  # [19, ]
        for tissue_type, tid in self.tissue_types.items():
            tissue_bpq_scores[tid] = np.nanmean(tissue_binary_pq_scores[tid])

        tissue_bPQ = np.nanmean(tissue_bpq_scores)

        tissue_metrics = {}
        for tissue_type, tid in self.tissue_types.items():
            tissue_metrics[f"mPQ_{tissue_type}"] = tissue_mpq_scores[tid]
            tissue_metrics[f"bPQ_{tissue_type}"] = tissue_bpq_scores[tid]

        metrics = {"mPQ": tissue_mPQ, "bPQ": tissue_bPQ}
        # metrics.update(tissue_metrics)
        df = pd.DataFrame({"mPQ": tissue_mpq_scores, "mBQ": tissue_bpq_scores})
        df.to_csv("metrics.csv", index=False)
        return metrics

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

    def prepare_test_img(self, idx):
        # 和train一样，获取gt
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def result_to_inst(self, result):
        return result["inst_preds"].astype("int32")

    def format_results(self, results, **kwargs):
        for file_idx, res in enumerate(results):
            pred = self.result_to_inst(res)
            pred, mapping = renumber(pred)
            binary_map = pred.copy()
            binary_map[binary_map > 0] = 1
            label_idx = np.unique(pred)  # 0,1,2,...N
            print(f"length of label_idx:{len(label_idx)}, max id is: {np.max(label_idx)}.")

            # 根据mapping调整res键的编号
            for key, value in res.items():
                if key != 'inst_preds':
                    modified = {mapping.get(k, k): v for k, v in value.items()}
                    res[key] = modified

            # [(y,x), ...]
            inst_centroid_yx = measurements.center_of_mass(
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

            inst_id_list = np.unique(pred)[1:].tolist()  # exclude background
            inst_info_dict = {}
            args = [(pred, inst_id, res) for inst_id in inst_id_list]
            print(datetime.now())

            # 8进程
            with Pool(8) as pool:
                results = list(tqdm(pool.starmap(get_inst_info, args),
                                    total=len(inst_id_list)))

            for inst_id, inst_info in results:
                if inst_id is not None and inst_info is not None:
                    inst_info_dict[inst_id] = inst_info

            # for inst_id in tqdm(inst_id_list):
            #     inst_map = pred == inst_id
            #     # TODO: change format of bbox output
            #     rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            #     inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            #     inst_map = inst_map[
            #                inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
            #                ]
            #     inst_map = inst_map.astype(np.uint8)
            #     inst_moment = cv2.moments(inst_map)
            #     inst_contour = cv2.findContours(
            #         inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            #     )
            #     # * opencv protocol format may break
            #     inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            #     # < 3 points dont make a contour, so skip, likely artifact too
            #     # as the contours obtained via approximation => too small or sthg
            #     if inst_contour.shape[0] < 3:
            #         continue
            #     if len(inst_contour.shape) != 2:
            #         continue  # ! check for trickery shape
            #     inst_centroid = [
            #         (inst_moment["m10"] / inst_moment["m00"]),
            #         (inst_moment["m01"] / inst_moment["m00"]),
            #     ]
            #     inst_centroid = np.array(inst_centroid)
            #     inst_contour[:, 0] += inst_bbox[0][1]  # X
            #     inst_contour[:, 1] += inst_bbox[0][0]  # Y
            #     inst_centroid[0] += inst_bbox[0][1]  # X
            #     inst_centroid[1] += inst_bbox[0][0]  # Y
            #     inst_info_dict[int(inst_id)] = {  # inst_id should start at 1
            #         "bbox": inst_bbox,
            #         "centroid": inst_centroid,
            #         "contour": inst_contour,
            #         "type_prob": None,
            #         "type": None,
            #     }
            print(datetime.now())
            self.__save_json(
                os.path.join(
                    kwargs["imgfile_prefix"],
                    img_info["filename"].split(".")[0] + ".json",
                ),
                inst_info_dict,
            )
            img = cv2.imread(os.path.join(self.img_dir, img_info["filename"]))
            overlay = self.visualize_instances_dict(img, inst_info_dict, False, self.type_colors)
            cv2.imwrite(
                os.path.join(kwargs["imgfile_prefix"], img_info["filename"]), overlay
            )
