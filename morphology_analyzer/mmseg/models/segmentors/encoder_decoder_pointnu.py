# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight
from collections import Counter
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
import mmcv
from transformers.models.sam.modeling_sam import SamVisionEncoderOutput

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from ..utils import matrix_nms


@SEGMENTORS.register_module()
class EncoderDecoderPointNu(EncoderDecoder):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(EncoderDecoderPointNu, self).__init__(
            backbone,
            decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        seg_weight=None,
        return_feat=False,
        gt_heat_map=None,
        gt_inst_mask=None,
        gt_is_center=None,
    ):
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses["features"] = x

        loss_decode = self._decode_head_forward_train(
            x,
            img_metas,
            gt_semantic_seg,
            seg_weight,
            gt_heat_map=gt_heat_map,
            gt_inst_mask=gt_inst_mask,
            gt_is_center=gt_is_center,
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight
            )
            losses.update(loss_aux)

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if isinstance(out, torch.Tensor):
            out = resize(
                input=out,
                size=img.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return out

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        output_dict = self.inference(img, img_meta, rescale)

        output_dict["inst_preds"] = output_dict["inst_preds"].cpu().numpy()
        return output_dict

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            output_dict = self.slide_inference(img, img_meta, rescale)
        else:
            # seg_logit = self.whole_inference(img, img_meta, rescale)
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

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        min_px = self.test_cfg.get("min_px", 16)

        pad_t = h_crop - h_stride
        pad_b = pad_t
        pad_l = w_crop - w_stride
        pad_r = pad_l
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        inst_id = 1

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
                # for channel in range(crop_img.shape[1]):
                #     img_c = crop_img[:, channel, :, :]
                #     i99 = torch.quantile(img_c, 0.99)
                #     i1 = torch.quantile(img_c, 0.01)
                #     if i99 - i1 > 1e-3:
                #         norm_img_c = (img_c - i1) / (i99 - i1)
                #         crop_img[:, channel, :, :] = norm_img_c
                #         # self.mean[channel] = i1
                #         # self.std[channel] = i99 - i1
                #     else:
                #         crop_img[:, channel, :, :] = 0.
                #         # self.mean[channel] = (i1 + i99) / 2
                #         # self.std[channel] = 1

                feature_preds, kernel_preds, heat_preds = self.encode_decode(
                    crop_img, img_meta
                )

                # seg_masks [num_instance, 256, 256]
                # inst_types [num_instance, ]
                # heat_scores [num_instance, ]
                seg_masks, inst_types, heat_scores = self.postprocess(
                    feature_preds=feature_preds,
                    kernel_preds=kernel_preds,
                    heat_preds=heat_preds,
                )
                if seg_masks is None:
                    continue
                for idx in range(seg_masks.shape[0]):
                    seg_ = seg_masks[idx]
                    label_ = int(inst_types[idx])
                    score_ = float(heat_scores[idx])
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

    def postprocess(
        self,
        feature_preds,
        kernel_preds,
        heat_preds,
        use_nms=True,
        nms_pre=500,
        max_per_img=200,
        score_thr=0.4,
        update_thr=0.2,
    ):
        N, E, h, w = feature_preds.shape
        assert N == 1
        num_class = heat_preds.shape[1] + 1

        kernel_preds = kernel_preds[0].permute(1, 2, 0).view(-1, E)
        heat_preds = (
            self.points_nms(heat_preds.sigmoid(), kernel=3)
            if not use_nms
            else heat_preds.sigmoid()
        )
        heat_preds = heat_preds.permute(0, 2, 3, 1).view(-1, num_class - 1)
        inds = heat_preds > score_thr
        heat_scores = heat_preds[inds]

        inds = inds.nonzero(as_tuple=False)
        inst_types = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        I, N = kernel_preds.shape
        if I == 0:
            return [None] * 3
        kernel_preds = kernel_preds.view(I, N, 1, 1)

        seg_preds = F.conv2d(feature_preds, kernel_preds, stride=1).squeeze(0)
        seg_masks = seg_preds.sigmoid() > 0.5  # (N, 64 ** 2, h, w)
        sum_masks = seg_masks.sum((1, 2)).float()

        seg_scores = (seg_preds.sigmoid() * seg_masks.float()).sum((1, 2)) / sum_masks
        heat_scores *= seg_scores

        # min area filter.

        keep = sum_masks > 4
        if keep.sum() == 0:
            return [None] * 3
        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        heat_scores = heat_scores[keep]
        inst_types = inst_types[keep]
        if use_nms:
            # sort and keep top nms_pre
            sort_inds = torch.argsort(heat_scores, descending=True)
            if len(sort_inds) > nms_pre:
                sort_inds = sort_inds[:nms_pre]
            seg_masks = seg_masks[sort_inds, :, :]  # [num, 256, 256]
            seg_preds = seg_preds[sort_inds, :, :]  # [num, 256, 256]
            sum_masks = sum_masks[sort_inds]  # [num, ]
            heat_scores = heat_scores[sort_inds]  # [num, ]
            inst_types = inst_types[sort_inds]  # [num, ] 从0开始

            heat_scores = matrix_nms(
                seg_masks,
                inst_types,
                heat_scores,
                kernel="gaussian",
                sigma=10,
                sum_masks=sum_masks,
            )
            keep = heat_scores >= update_thr
            if keep.sum() == 0:
                return [None] * 3
            seg_preds = seg_preds[keep, :, :]
            heat_scores = heat_scores[keep]
            inst_types = inst_types[keep]

            # sort and keep top_k
            sort_inds = torch.argsort(heat_scores, descending=True)
            if len(sort_inds) > max_per_img:
                sort_inds = sort_inds[:max_per_img]
            seg_preds = seg_preds[sort_inds, :, :]
            heat_scores = heat_scores[sort_inds]
            inst_types = inst_types[sort_inds]

        seg_masks = seg_preds.sigmoid() > 0.5
        return seg_masks, inst_types, heat_scores

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

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        total_seg_masks = []
        total_inst_labels = []
        total_inst_scores = []
        total_sum_masks = []
        for i in range(0, len(imgs)):
            output_dict = self.inference(imgs[i], img_metas[i], rescale)
            inst_preds = output_dict["inst_preds"]
            inst_score = output_dict["inst_score"]
            inst_labels = output_dict["inst_labels"]

            for inst_id in torch.unique(inst_preds):
                if inst_id == 0:
                    continue
                tmp_mask = torch.zeros_like(inst_preds, dtype=torch.int32)
                tmp_mask[inst_preds == inst_id] = 1
                total_seg_masks.append(tmp_mask)
                total_inst_labels.append(inst_labels[int(inst_id)])
                total_inst_scores.append(inst_score[int(inst_id)])
                total_sum_masks.append(tmp_mask.sum())

        total_seg_masks = torch.stack(total_seg_masks, dim=0)
        total_inst_labels = torch.tensor(total_inst_labels)
        total_inst_scores = torch.tensor(total_inst_scores)
        total_sum_masks = torch.stack(total_sum_masks, dim=0)

        total_inst_scores = matrix_nms(
            total_seg_masks,
            total_inst_labels,
            total_inst_scores,
            kernel="gaussian",
            sigma=10,
            sum_masks=total_sum_masks,
        )
        keep = total_inst_scores >= 0.2
        if keep.sum() == 0:
            return output_dict

        total_seg_masks = total_seg_masks[keep]
        total_inst_labels = total_inst_labels[keep]
        total_inst_scores = total_inst_scores[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(total_inst_scores, descending=True)
        max_per_img = 1000
        if len(sort_inds) > max_per_img:
            sort_inds = sort_inds[:max_per_img]
        total_seg_masks = total_seg_masks[sort_inds, :, :]
        total_inst_labels = total_inst_labels[sort_inds]
        total_inst_scores = total_inst_scores[sort_inds]

        inst_preds = torch.zeros(total_seg_masks.shape[1:], dtype=torch.int32)
        inst_score = {}
        inst_labels = {}
        inst_id = 1
        for i in range(total_seg_masks.shape[0]):
            seg_mask = total_seg_masks[-i - 1]
            inst_preds[seg_mask == 1] = inst_id
            inst_score[inst_id] = total_inst_scores[-i - 1]
            inst_labels[inst_id] = total_inst_labels[-i - 1]
            inst_id += 1

        return {
            "inst_preds": inst_preds.cpu().numpy(),
            "inst_score": inst_score,
            "inst_labels": inst_labels,
        }

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if isinstance(x, SamVisionEncoderOutput):
            x = x[1]
        if self.with_neck:
            x = self.neck(x)
        return x