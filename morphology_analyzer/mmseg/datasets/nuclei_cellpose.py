# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
from torch.utils.data import Dataset

from . import CityscapesDataset

import os
import numpy as np
import cv2
import os.path as osp
from collections import OrderedDict
from functools import reduce
import warnings
import scipy
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter1d
import fastremap

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

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
)
from .nuclei import NucleiDataset


def result_to_inst(
    result, p=None, niter=200, interp=True, flow_threshold=0.4, min_size=15
):
    """
    网络输出转换成instance map

    Args:
        result: results (torch.Tensor): 3通道，前两个通道是vec(梯度)，最后一个通道是语义分割(未做sigmoid)

    Returns:

    """
    # h, w
    cellprob = result[2, ...]
    # 2, h, w
    dp = result[:2, ...]
    cp_mask = cellprob > 0.0
    if np.any(cp_mask):
        if p is None:
            p, inds = follow_flows(dp * cp_mask / 5.0, niter=niter, interp=interp)
            if inds is None:
                shape = cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p

        # calculate masks
        mask = get_masks(p, iscell=cp_mask)
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(mask, dp, threshold=flow_threshold)
    else:
        shape = cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p

    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
    return mask, p


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


@njit("(float32[:,:,:], float32[:,:,:], int32[:,:], int32)", nogil=True)
def steps2D(p, dP, inds, niter):
    """run dynamics of pixels to recover masks in 2D

    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]
            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))
    return p


def steps2D_interp(p, dP, niter):
    shape = dP.shape[1:]
    dPt = np.zeros(p.shape, np.float32)

    for t in range(niter):
        map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
        for k in range(len(p)):
            p[k] = np.minimum(shape[k] - 1, np.maximum(0, p[k] + dPt[k]))
    return p


@njit(
    [
        "(int16[:,:,:], float32[:], float32[:], float32[:,:])",
        "(float32[:,:,:], float32[:], float32[:], float32[:,:])",
    ],
    cache=True,
)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y

    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C, Ly, Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly - 1, max(0, yc_floor[i]))
        xf = min(Lx - 1, max(0, xc_floor[i]))
        yf1 = min(Ly - 1, yf + 1)
        xf1 = min(Lx - 1, xf + 1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c, i] = (
                np.float32(I[c, yf, xf]) * (1 - y) * (1 - x)
                + np.float32(I[c, yf, xf1]) * (1 - y) * x
                + np.float32(I[c, yf1, xf]) * y * (1 - x)
                + np.float32(I[c, yf1, xf1]) * y * x
            )


def get_masks(p, iscell=None, rpad=20):
    """create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(
                np.arange(shape0[0]),
                np.arange(shape0[1]),
                np.arange(shape0[2]),
                indexing="ij",
            )
        elif dims == 2:
            inds = np.meshgrid(
                np.arange(shape0[0]), np.arange(shape0[1]), indexing="ij"
            )
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0


def remove_bad_flow_masks(masks, flows, threshold=0.4):
    """remove masks which have inconsistent flows

    Uses metrics.flow_error to compute flows from predicted masks
    and compare flows to predicted flows from network. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    merrors, _ = flow_error(masks, flows)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def flow_error(maski, dP_net):
    """error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------

    maski: ND-array (int)
        masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float)
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks

    """
    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows(maski)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += ndimage.mean(
            (dP_masks[i] - dP_net[i] / 5.0) ** 2,
            maski,
            index=np.arange(1, maski.max() + 1),
        )

    return flow_errors, dP_masks


def masks_to_flows(masks):
    """convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """
    if masks.max() == 0:
        return np.zeros((2, *masks.shape), "float32")

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)

    centers = np.zeros((masks.max(), 2), "int")
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices]
    )
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter)

    # normalize
    mu /= 1e-20 + (mu**2).sum(axis=0) ** 0.5

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0), dtype=np.float32)
    mu0[:, y - 1, x - 1] = mu
    mu_c = np.zeros_like(mu0)

    # _, ax = plt.subplots(2, 2, figsize=(11, 11))
    # ax[0][0].imshow(((mu0[0] + 1) * 127).astype(np.uint8), cmap="gray")
    # ax[0][1].imshow(((mu0[1] + 1) * 127).astype(np.uint8), cmap="gray")
    # ax[1][0].imshow(masks)
    # plt.show()
    return mu0


def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200):
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


@DATASETS.register_module()
class NucleiCellPoseDataset(NucleiDataset):
    CLASSES = ("Nuclei",)
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(NucleiCellPoseDataset, self).__init__(**kwargs)

    def get_gt_instance_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                "DeprecationWarning: ``efficient_test`` has been deprecated "
                "since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory "
                "friendly by default. "
            )

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results["gt_instance"]

    def evaluate(
        self, results, metric="mIoU", logger=None, gt_instance_maps=None, **kwargs
    ):
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
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_instance_maps is None:
                gt_instance_maps = self.get_gt_instance_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = self.run_nuclei_inst_stat(results, gt_instance_maps)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
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

    def run_nuclei_inst_stat(self, results, gt_instance_maps):
        """
        计算各种指标
        Args:
            results: results (list[torch.Tensor]): 3通道，前两个通道是vec(梯度)，最后一个通道是语义分割(未做sigmoid)
            gt_instance_maps: 生成器

        Returns:

        """
        metrics = [[], [], [], [], [], []]
        for res, gt_ins_map in zip(results, gt_instance_maps):
            pred = result_to_inst(res)[0]

            # to ensure that the instance numbering is contiguous
            pred = remap_label(pred, by_size=False)
            gt_ins_map = remap_label(gt_ins_map, by_size=False)

            pq_info = get_fast_pq(gt_ins_map, pred, match_iou=0.5)[0]
            metrics[0].append(get_dice_1(gt_ins_map, pred))
            if pred.max() == 0:
                metrics[1].append(0.0)  # avoid some bug
            else:
                metrics[1].append(get_fast_aji(gt_ins_map, pred))
            metrics[2].append(pq_info[0])  # dq
            metrics[3].append(pq_info[1])  # sq
            metrics[4].append(pq_info[2])  # pq
            metrics[5].append(get_fast_aji_plus(gt_ins_map, pred))

        ####
        metrics = np.array(metrics)
        metrics_avg = np.mean(metrics, axis=-1)
        ret_metrics = {
            "DICE": metrics_avg[0],
            "AJI": metrics_avg[1],
            "DQ": metrics_avg[2],
            "SQ": metrics_avg[3],
            "PQ": metrics_avg[4],
            "AJI_PLUS": metrics_avg[5],
        }

        return ret_metrics


def follow_flows(dP, mask=None, niter=200, interp=True):
    """define pixels and run dynamics to recover masks in 2D

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D)
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, None

    if not interp:
        p = steps2D(p, dP.astype(np.float32), inds, niter)

    else:
        p_interp = steps2D_interp(p[:, inds[:, 0], inds[:, 1]], dP, niter)
        p[:, inds[:, 0], inds[:, 1]] = p_interp
    return p, inds
