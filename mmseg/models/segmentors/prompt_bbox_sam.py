# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight
import math
from collections import Counter
from pathlib import Path

from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoder,
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamModel,
    SamVisionEncoderOutput,
)
import mmcv
from mmcv.runner import load_checkpoint
import cv2
import numpy as np
from scipy import ndimage

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from ..utils import matrix_nms
from .sam import SAM


@SEGMENTORS.register_module()
class PromptBBoxSam(BaseSegmentor):
    def __init__(
        self,
        prompter,
        sam_model,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        pretrained=None,
    ):
        super(PromptBBoxSam, self).__init__(init_cfg)

        # build SAM
        self.sam = SAM(
            hf_pretrain_name=sam_model.hf_pretrain_name, init_cfg=sam_model.init_cfg
        )

        # build prompter
        self.prompter = builder.build_backbone(prompter)  # 输入图片，输出一堆坐标点

        self.num_classes = self.prompter.num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dice_loss = DiceLoss("binary")
        self.focal_loss = BinaryFocalLoss()

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(
        self,
        img,
        img_metas,
        gt_inst_mask,
        gt_bboxes,
        gt_sam_inst_masks,
        # gt_sam_prompt_points,
        # gt_sam_prompt_labels,
        gt_sam_prompt_bboxes,
        seg_weight=None,
        **kwargs,
    ):
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值

        out_dict = self.prompter.forward_train(
            img, img_metas, gt_bboxes, gt_bboxes_ignore=None
        )

        losses = dict()
        losses.update(out_dict["loss"])

        cell_num = torch.tensor(
            [pp.shape[0] for pp in gt_sam_prompt_bboxes], device=img.device
        )
        # gt_sam_prompt_points = torch.cat(gt_sam_prompt_points, dim=0).unsqueeze(1)
        # gt_sam_prompt_labels = torch.cat(gt_sam_prompt_labels, dim=0).unsqueeze(1)
        gt_sam_prompt_bboxes = torch.cat(gt_sam_prompt_bboxes, dim=0).unsqueeze(1)

        if gt_sam_prompt_bboxes.shape[0] == 0:
            return losses

        sam_output = self.sam(
            img,
            input_boxes=gt_sam_prompt_bboxes,
            multimask_output=False,
            cell_num=cell_num,
        )
        sam_seg = sam_output[1].squeeze(1)  # [num, 1, 64, 64]
        # resize to gt_inst_mask size
        sam_seg = F.interpolate(
            sam_seg,
            size=gt_inst_mask[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        gt_sam_inst_masks = torch.cat(gt_sam_inst_masks, dim=0).long()

        loss_dice = self.dice_loss(sam_seg, gt_sam_inst_masks)
        loss_focal = self.focal_loss(sam_seg, gt_sam_inst_masks.unsqueeze(1))

        losses["sam.loss_dice"] = loss_dice
        losses["sam.loss_focal"] = loss_focal

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        min_px = self.test_cfg.get("min_px", 16)  # 过滤小面积
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值

        pad_t = h_crop - h_stride
        pad_b = pad_t
        pad_l = w_crop - w_stride
        pad_r = pad_l
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        inst_id = 1

        # gt的prompt也要相应位移
        if "gt_sam_prompt_bboxes" in kwargs:
            assert len(kwargs["gt_sam_prompt_bboxes"]) == 1
            kwargs["gt_sam_prompt_bboxes"][0][..., 0] += pad_l
            kwargs["gt_sam_prompt_bboxes"][0][..., 1] += pad_t
            kwargs["gt_sam_prompt_bboxes"][0][..., 2] += pad_l
            kwargs["gt_sam_prompt_bboxes"][0][..., 3] += pad_t

        batch_size, _, h_img, w_img = img.size()
        assert batch_size == 1
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        inst_preds = img.new_zeros((h_img, w_img)).type(torch.int32)
        inst_score = {}
        inst_labels = {}
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pred_scores_total, sam_seg_total = self.encode_decode(
                    crop_img,
                    img_meta,
                    h_idx=h_idx,
                    w_idx=w_idx,
                    y1=y1,
                    x1=x1,
                    y2=y2,
                    x2=x2,
                    **kwargs,
                )

                if len(pred_scores_total) == 0:
                    continue

                pred_scores = pred_scores_total[0]  # [num, ] prompter输出的概率
                sam_seg = sam_seg_total[0]  # [num, 256, 256] SAM输出, before sigmoid

                # 过滤，nms
                # seg_masks: [num, 256, 256] 0/1矩阵
                # inst_types: [num, ] 目标类型，没有指定则为全0
                # pred_scores: [num, ] 目标得分
                seg_masks, inst_types, pred_scores = self.postprocess(
                    pred_scores, sam_seg
                )
                if seg_masks is None:
                    continue
                for idx in range(seg_masks.shape[0]):
                    seg_ = seg_masks[idx]
                    label_ = int(inst_types[idx])
                    score_ = float(pred_scores[idx])
                    center_w, center_h, width, height = self.get_ins_info(
                        seg_.cpu().numpy(), method="bbox"
                    )
                    center_h = math.ceil(center_h)
                    center_w = math.ceil(center_w)
                    # 只保留中间的预测，因为靠近边缘的预测可能不完整，且其他patch有完整的预测
                    if (
                        center_h >= h_crop // 2 - h_stride // 2
                        and center_h <= h_crop // 2 + h_stride // 2
                        and center_w >= w_crop // 2 - w_stride // 2
                        and center_w <= w_crop // 2 + w_stride // 2
                    ):
                        focus_area = inst_preds[y1:y2, x1:x2].clone()

                        # 判断有没有重叠
                        if torch.sum(torch.logical_and(focus_area > 0, seg_)) == 0:
                            # 没有重叠
                            inst_preds[y1:y2, x1:x2] = torch.where(
                                focus_area > 0,
                                focus_area,
                                seg_.type(torch.int32) * (inst_id),
                            )
                            inst_score[inst_id] = score_
                            inst_labels[inst_id] = label_
                            inst_id += 1
                        else:
                            # 有重叠，找到重叠的inst_id
                            compared_num, _ = Counter(
                                (focus_area * seg_).flatten().tolist()
                            ).most_common(2)[1]
                            if compared_num == 0:
                                compared_num, _ = Counter(
                                    (focus_area * seg_).flatten().tolist()
                                ).most_common(2)[0]
                            assert compared_num > 0
                            compared_num = int(compared_num)
                            compared_score = inst_score[compared_num]
                            if (
                                torch.sum(
                                    torch.logical_and(focus_area == compared_num, seg_)
                                )
                                / torch.sum(
                                    torch.logical_or(focus_area == compared_num, seg_)
                                )
                                > 0.7
                            ):  # IoU>0.7判断重叠
                                if compared_score > score_:
                                    pass
                                else:
                                    focus_area[focus_area == compared_num] = 0
                                    inst_preds[y1:y2, x1:x2] = focus_area
                                    inst_preds[y1:y2, x1:x2] = torch.where(
                                        focus_area > 0,
                                        focus_area,
                                        seg_.type(torch.int32) * (inst_id),
                                    )
                                    inst_score[inst_id] = score_
                                    inst_labels[inst_id] = label_
                                    inst_id += 1
                            else:
                                inst_preds[y1:y2, x1:x2] = torch.where(
                                    focus_area > 0,
                                    focus_area,
                                    seg_.type(torch.int32) * (inst_id),
                                )
                                inst_score[inst_id] = score_
                                inst_labels[inst_id] = label_
                                inst_id += 1

        inst_preds = inst_preds[pad_t:-pad_b, pad_l:-pad_r]
        for ui in torch.unique(inst_preds):
            if ui == 0:
                continue
            if torch.sum(inst_preds == ui) < min_px:
                inst_preds[inst_preds == ui] = 0

        if rescale:
            inst_preds = mmcv.imresize(
                inst_preds.cpu().numpy(),
                img_meta[0]["ori_shape"][:2],
                interpolation="nearest",
            )

        return {
            "inst_preds": torch.from_numpy(inst_preds),
            "inst_score": inst_score,
            "inst_labels": inst_labels,
        }

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]["ori_shape"][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )

        return seg_logit

    def inference(self, img, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            output_dict = self.slide_inference(img, img_meta, rescale, **kwargs)
        else:
            raise ValueError("only support slide mode in test")
        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output_dict["inst_preds"] = output_dict["inst_preds"].flip(dims=(1,))
            elif flip_direction == "vertical":
                output_dict["inst_preds"] = output_dict["inst_preds"].flip(dims=(0,))

        return output_dict

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        assert rescale is True
        output_dict = self.inference(img, img_meta, rescale, **kwargs)

        output_dict["inst_preds"] = output_dict["inst_preds"].cpu().numpy()
        return output_dict

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    def encode_decode(self, img, img_metas, **kwargs):
        y1 = kwargs["y1"]
        x1 = kwargs["x1"]
        y2 = kwargs["y2"]
        x2 = kwargs["x2"]
        gt_sam_prompt_bboxes = kwargs.get("gt_sam_prompt_bboxes", None)
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值

        if gt_sam_prompt_bboxes is not None:
            # [n, 1, 4]
            gt_sam_prompt_bboxes = (
                torch.cat(gt_sam_prompt_bboxes, dim=0).squeeze(0).unsqueeze(1)
            )
            # [n, ]
            indexes = torch.where(
                (gt_sam_prompt_bboxes[..., 0, 0] > x1)
                & (gt_sam_prompt_bboxes[..., 0, 2] < x2)
                & (gt_sam_prompt_bboxes[..., 0, 1] > y1)
                & (gt_sam_prompt_bboxes[..., 0, 3] < y2)
            )
            gt_sam_prompt_bboxes = gt_sam_prompt_bboxes[indexes]  # [n, 1, 4]
            gt_sam_prompt_bboxes[..., 0] -= x1
            gt_sam_prompt_bboxes[..., 2] -= x1
            gt_sam_prompt_bboxes[..., 1] -= y1
            gt_sam_prompt_bboxes[..., 3] -= y1
            cell_num = gt_sam_prompt_bboxes.shape[0]
            gt_sam_prompt_bboxes = torch.cat(
                [
                    gt_sam_prompt_bboxes,
                    torch.ones((cell_num, 1, 1), device=gt_sam_prompt_bboxes.device),
                ],
                dim=2,
            )
            gt_sam_prompt_bboxes = gt_sam_prompt_bboxes.squeeze(1)  # 对齐后面
        else:
            proposal_list = self.prompter.forward_test(img, img_metas)

        sam_seg_total = []
        pred_scores_total = []
        for batch_idx in range(img.shape[0]):
            if gt_sam_prompt_bboxes is not None:
                proposal = gt_sam_prompt_bboxes
            else:
                proposal = proposal_list[batch_idx]

            if len(proposal) > 0:
                sam_output = self.sam(
                    img[batch_idx].unsqueeze(0),
                    input_boxes=proposal.unsqueeze(1)[:, :, :4],
                    multimask_output=False,
                    cell_num=len(proposal),
                )
            else:
                continue

            sam_seg = sam_output[1].squeeze(1)  # [num_gt, 1, 64, 64]
            # resize to gt_inst_mask size
            sam_seg = F.interpolate(
                sam_seg,
                size=img[batch_idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            sam_seg_total.append(sam_seg)
            pred_scores_total.append(proposal[:, -1])

        return pred_scores_total, sam_seg_total

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
            return (
                float(center_w),
                float(center_h),
                float(EC_radius * 2),
                float(EC_radius * 2),
            )
        elif method == "bbox":
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(
                np.array(seg_mask).astype(np.uint8)
            )
            center_w = bbox_x + bbox_w / 2
            center_h = bbox_y + bbox_h / 2
            return float(center_w), float(center_h), float(bbox_w), float(bbox_h)
        elif method == "area":
            center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
            equal_diameter = (np.sum(seg_mask) / 3.1415) ** 0.5 * 2
            return (
                float(center_w),
                float(center_h),
                float(equal_diameter),
                float(equal_diameter),
            )
        else:
            raise NotImplementedError

    def postprocess(self, pred_scores, sam_seg):
        nms_pre = self.test_cfg.get("nms_pre", 500)  # nms前的目标数
        max_per_img = self.test_cfg.get("max_per_img", 200)  # nms后的目标数
        update_thr = self.test_cfg.get("update_thr", 0.2)  # nms后score阈值

        seg_masks = sam_seg.sigmoid() > 0.5  # [num, 256, 256] 0/1 mask
        sum_masks = seg_masks.sum((1, 2)).float()  # [num_gt, ] 每个mask的面积
        # [num, ] 每个mask的平均概率
        seg_scores = (sam_seg.sigmoid() * seg_masks).sum((1, 2)) / (sum_masks + 1e-6)

        pred_scores *= seg_scores  # prompter输出的概率*每个mask的平均概率

        # min area filter.
        keep = sum_masks > 4
        if keep.sum() == 0:
            return [None] * 3

        seg_masks = seg_masks[keep, ...]  # [num, 256, 256] 0/1 mask
        sam_seg = sam_seg[keep, ...]  # [num, 256, 256] SAM输出, before sigmoid
        sum_masks = sum_masks[keep]  # [num_gt, ] 每个mask的面积
        pred_scores = pred_scores[keep]  # prompter输出的概率*每个mask的平均概率

        # nms
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > nms_pre:
            sort_inds = sort_inds[:nms_pre]

        seg_masks = seg_masks[sort_inds, :, :]  # [num, 256, 256]
        sam_seg = sam_seg[sort_inds, :, :]  # [num, 256, 256]
        sum_masks = sum_masks[sort_inds]  # [num, ]
        pred_scores = pred_scores[sort_inds]  # [num, ]
        inst_types = torch.zeros_like(
            pred_scores, dtype=torch.int32, device=pred_scores.device
        )  # [num, ] 从0开始

        pred_scores = matrix_nms(
            seg_masks,
            inst_types,
            pred_scores,
            kernel="gaussian",
            sigma=10,
            sum_masks=sum_masks,
        )
        keep = pred_scores >= update_thr
        if keep.sum() == 0:
            return [None] * 3
        sam_seg = sam_seg[keep, :, :]
        pred_scores = pred_scores[keep]
        inst_types = inst_types[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > max_per_img:
            sort_inds = sort_inds[:max_per_img]
        sam_seg = sam_seg[sort_inds, :, :]
        pred_scores = pred_scores[sort_inds]
        inst_types = inst_types[sort_inds]

        seg_masks = sam_seg.sigmoid() > 0.5

        return seg_masks, inst_types, pred_scores
