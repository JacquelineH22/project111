import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import json
import torch

from .stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)


def run_nuclei_type_stat(pred_dir, true_dir, binary=False, ext="pth", merge=False, type_uid_list=None, exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
                     
    """
    file_list = glob.glob(pred_dir + "*.json")
    file_list.sort()  # ensure same order [1]

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        if ext == "mat":
            true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        elif ext == "pth":
            true_info = torch.load(os.path.join(true_dir, basename + ".pth"))
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        # * for converting the GT type in CoNSeP
        if merge:
            true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
            true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4
            
        # for binary semantic seg
        if binary:
            true_inst_type[true_inst_type > 0] = 1

        # 读取 JSON 文件
        with open(file_list[file_idx], 'r') as f:
            pred_info = json.load(f)
        
        pred_info = pred_info['nuc']
        # dont squeeze, may be 1 instance exist
        pred_centroid = np.empty((0, 2)).astype("float32")
        pred_inst_type = np.empty((0, 1)).astype("int32")
        for id in pred_info.keys():
            pred_centroid = np.vstack((pred_centroid, pred_info[id]['centroid'])).astype("float32")
            new_type = pred_info[id]['type_prob'].index(max(pred_info[id]['type_prob'][1:]))
            pred_inst_type = np.vstack((pred_inst_type, new_type)).astype("int32")
            # pred_inst_type = np.vstack((pred_inst_type, pred_info[id]['type'])).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
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

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

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

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()
        if (0 in type_uid_list):
            type_uid_list = type_uid_list[1:]  # 去掉0

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(np.array(results_list))
    return


def run_nuclei_inst_stat(pred_dir, true_dir, ext="mat", print_img_stats=False):
    # print stats of each image
    print(pred_dir)

    file_list = glob.glob("%s/*%s" % (pred_dir, ".npy"))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        # basename = filename.split(".")[0]
        basename = os.path.splitext(filename)[0]
        
        if ext == "mat":
            true = sio.loadmat(os.path.join(true_dir, basename + "."+ext))
            true = (true["inst_map"]).astype("int32")
        elif ext == "tif":
            true = cv2.imread(os.path.join(true_dir, basename.replace("img","instancemask") + "."+ext),
                              cv2.IMREAD_UNCHANGED)
        elif ext == "pth":
            true = torch.load(os.path.join(true_dir, basename + "."+ext))
            true = (true["inst_map"]).astype("int32")

        pred = np.load(os.path.join(pred_dir, basename + ".npy"))
        pred = pred.astype("int32")
        if(np.all(true==0) or np.all(pred==0)):
            continue

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics

def inst_to_type(inst_map, type_map):
        inst_to_type = {}
        for inst_id in np.unique(inst_map):
            if inst_id == 0: 
                continue
            mask = (inst_map == inst_id)
            # 找到该实例区域内最多出现的类型标签
            type_label = np.bincount(type_map[mask].astype(int)).argmax()
            inst_to_type[inst_id] = type_label
        
        return inst_to_type

def calculate_centroid(mask):
    y_indices, x_indices = np.nonzero(mask)
    
    # 计算质心
    total_weight = len(x_indices)  # 质心的权重是非零像素的数量（对于二值掩膜，权重是1）
    if total_weight == 0:
        return None 

    # 计算x和y方向的质心坐标
    cx = np.sum(x_indices) / total_weight
    cy = np.sum(y_indices) / total_weight
    
    return (cx, cy)

def run_nuclei_inst_stat_tissue(pred_dir, true_dir, ext="npy"):
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

    num_classes = 6
    tissue_types = {
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

    
    file_list = glob.glob("%s/*%s" % (pred_dir, "."+ext))
    file_list.sort()  # ensure same order
    nuclei_pq_scores = []  # [2722, 5]
    # [19, n, 5] 19类组织，每类组织n张图(不同组织n不相等)，每张图5类的pq
    tissue_nuclei_pq_scores = [[] for _ in tissue_types]
    binary_pq_scores = []  # [2722,] image_id
    tissue_binary_pq_scores = [[] for _ in tissue_types]  # [19, n]

    for filepath in file_list[:]:
        filename = os.path.basename(filepath)
        basename = filename.split(".")[0]

        gt = torch.load(os.path.join(true_dir, basename + ".pth"))
        true = (gt["inst_map"]).astype("int32")

        if ext == "npy":
            pred = np.load(os.path.join(pred_dir, basename + "."+ext))
            pred_inst_map = pred.astype("int32")
            # 读取 JSON 文件
            with open(os.path.join(pred_dir, basename + ".json"), 'r') as f:
                pred_info = json.load(f)       
            pred_info = pred_info['nuc']
            # dont squeeze, may be 1 instance exist
            pred_centroid = np.empty((0, 2)).astype("float32")
            pred_type = {}

            for id in pred_info.keys():
                pred_centroid = np.vstack((pred_centroid, pred_info[id]['centroid'])).astype("float32")
                pred_type[int(id)] = pred_info[id]['type']+1

        else:
            pred = sio.loadmat(os.path.join(pred_dir, basename))
            pred_inst_map = pred["inst_map"].astype("int32")
            # if pred.get("inst_centroid").any():
            if 'inst_centroid' in pred.keys():
                pred_centroid = pred["inst_centroid"].astype("float32")
            else:
                pred_centroid = np.empty((0, 2)).astype("float32")
                for id in np.unique(pred_inst_map):
                    if id == 0:
                        continue
                    centroid = calculate_centroid((pred_inst_map == id).astype(int))
                    pred_centroid = np.vstack((pred_centroid, centroid)).astype("float32")
            if "inst_type" in pred.keys():
                pred_type = {i+1: pred["inst_type"][i,0] for i in range(len(pred["inst_type"]))}
            else:
                pred_type_map = pred["type_map"].astype("int32")
                pred_type = inst_to_type(pred_inst_map, pred_type_map)
        
        if(np.all(true==0) or np.all(pred==0)):
            continue

        # 每类的inst_map
        pred_inst_map_c = np.zeros((num_classes, *pred_inst_map.shape))
        for inst_id in np.unique(pred_inst_map):
            if inst_id == 0:
                continue
            else:
                ty = pred_type.get(inst_id, -1)
                pred_inst_map_c[ty, pred_inst_map == inst_id] = inst_id

        gt_inst_map = remap_label(gt["inst_map"], by_size=False)  # [256, 256]
        gt_type_map = gt["type_map"]  # [256, 256]
        gt_tissue_type = gt["tissue_type"]  # str

        # one-hot [256, 256, 6]
        gt_type_map = np.eye(6)[gt_type_map]
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
        tissue_binary_pq_scores[tissue_types[gt_tissue_type]].append(bpq_tmp)

        # 考虑类别的指标
        nuclei_type_pq = []  # 一张图5类分别的pq
        nuclei_type_dq = []
        nuclei_type_sq = []
        for c in range(num_classes):
            pred_nuclei_instance_class = remap_label(pred_inst_map_c[c])
            target_nuclei_instance_class = remap_label(
                gt_instance_types_nuclei[c]
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
        tissue_nuclei_pq_scores[tissue_types[gt_tissue_type]].append(
            nuclei_type_pq
        )

    tissue_mpq_scores = [0.0 for _ in tissue_types]  # [19, ]
    for tissue_type, tid in tissue_types.items():
        tmp = tissue_nuclei_pq_scores[tid]  # [num, 5]
        tissue_mpq_scores[tid] = np.nanmean(np.nanmean(tmp, axis=1))

    tissue_mPQ = np.nanmean(tissue_mpq_scores)

    tissue_bpq_scores = [0.0 for _ in tissue_types]  # [19, ]
    for tissue_type, tid in tissue_types.items():
        tissue_bpq_scores[tid] = np.nanmean(tissue_binary_pq_scores[tid])

    tissue_bPQ = np.nanmean(tissue_bpq_scores)

    tissue_metrics = {}
    for tissue_type, tid in tissue_types.items():
        tissue_metrics[f"mPQ_{tissue_type}"] = tissue_mpq_scores[tid]
        tissue_metrics[f"bPQ_{tissue_type}"] = tissue_bpq_scores[tid]

    metrics = {"mPQ": tissue_mPQ, "bPQ": tissue_bPQ}
    # metrics.update(tissue_metrics)
    df = pd.DataFrame({"mPQ": tissue_mpq_scores, "mBQ": tissue_bpq_scores})
    df.to_csv(f"maskrcnn_fold2test.csv", index=False)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the measurement,"
        "`type` for nuclei instance type classification or"
        "`instance` for nuclei instance segmentation or"
        "`tissue` for nuclei instance segmentation on tissue",
        nargs="?",
        default="instance",
        const="instance",
    )
    parser.add_argument(
        "--pred_dir", help="point to output dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--true_dir", help="point to ground truth dir", nargs="?", default="", const=""
    ),
    parser.add_argument(
        '--binary', action='store_true', default=False
    ),
    parser.add_argument(
        '--ext', type=str, default='mat'
    ),
    parser.add_argument(
        '--merge', action='store_true', default=False
    ),
    args = parser.parse_args()

    if args.mode == "instance":
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, args.ext)
    if args.mode == "type":
        run_nuclei_type_stat(args.pred_dir, args.true_dir, args.binary, args.ext, args.merge)
    if args.mode == "tissue":
        run_nuclei_inst_stat_tissue(args.pred_dir, args.true_dir, args.ext)
