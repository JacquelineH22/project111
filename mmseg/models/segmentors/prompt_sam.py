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
from mmcv.runner import force_fp32
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


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


@SEGMENTORS.register_module()
class PromptSam(BaseSegmentor):
    def __init__(
        self,
        prompter,
        sam_model,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        pretrained=None,
        num_neg_prompt=0,
        train_sam=True,
        train_prompt=True,
        heat_stride=4
    ):
        super(PromptSam, self).__init__(init_cfg)

        # build SAM
        if train_sam:
            self.sam = SAM(
                hf_pretrain_name=sam_model.hf_pretrain_name, init_cfg=sam_model.init_cfg
            )

        # build prompter
        if train_prompt:
            self.prompter = builder.build_backbone(prompter)  # 输入图片，输出一堆坐标点
                
            self.sam = SAM(
                hf_pretrain_name=sam_model.hf_pretrain_name, init_cfg=sam_model.init_cfg
            )
            for param in self.sam.parameters():
                param.requires_grad = False
            
            # 冻结整个 prompter 部分的参数
            # for param in self.prompter.parameters():
            #     param.requires_grad = False

            # # 只解冻 promter.head.cate_mask 的部分
            # for name, param in self.prompter.named_parameters():
            #     # if 'heads.head_mask' in name:
            #     if 'heads.head_cate' in name:
            #         print(name, param)
            #         param.requires_grad = True

        self.num_classes = prompter.num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dice_loss = DiceLoss("binary")
        self.focal_loss = BinaryFocalLoss()
        self.num_neg_prompt = num_neg_prompt
        self.train_sam = train_sam
        self.train_prompt = train_prompt
        self.heat_stride = heat_stride

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        gt_heat_map,
        gt_inst_mask,
        gt_is_center,
        center_xy,
        gt_sam_inst_masks,
        gt_sam_prompt_points,
        gt_sam_prompt_labels,
        seg_weight=None,
        **kwargs,
    ):
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值

        losses = dict()

        if self.train_prompt:
            out_dict = self.prompter.forward_train(
                img, img_metas, gt_heat_map, gt_semantic_seg=gt_semantic_seg,
                use_seg_loss = True
            )
            losses.update(out_dict["loss"])

        # ----------------------------------#
        # for idx in range(img.shape[0]):
        #     draw_img = img[idx].cpu().numpy()
        #     draw_img = np.ascontiguousarray(np.transpose(draw_img, (1, 2, 0)))
        #     mean = img_metas[idx]["img_norm_cfg"]["mean"].reshape(1, 1, 3)
        #     std = img_metas[idx]["img_norm_cfg"]["std"].reshape(1, 1, 3)
        #     draw_img = draw_img * std + mean
        #     draw_img = np.clip(draw_img, a_min=0, a_max=255).astype(np.uint8)
        #     for xy in center_xy[idx]:
        #         xy = (xy.cpu().numpy() / 4.0).astype(np.int32).tolist()
        #         draw_img = cv2.circle(draw_img, (xy[0], xy[1]), 3, (0, 255, 0), -1)
        #     cv2.imwrite(img_metas[idx]["ori_filename"], draw_img)
        # ----------------------------------#

        cell_num = torch.tensor(
            [pp.shape[0] for pp in gt_sam_prompt_points], device=img.device
        )
        gt_sam_prompt_points = torch.cat(gt_sam_prompt_points, dim=0).unsqueeze(1)
        # [num, 2, 1]  padding or neg
        gt_sam_prompt_labels = torch.cat(gt_sam_prompt_labels, dim=0).unsqueeze(1)  #list of [2,2]

        if not self.train_sam:
            return losses

        if gt_sam_prompt_points.shape[0] == 0:
            return losses

        sam_output = self.sam(
            img,
            gt_sam_prompt_points,
            gt_sam_prompt_labels,
            multimask_output=False,
            cell_num=cell_num,
        )
        # 随便写一个decoder head，使用sam_output中的image_embeddings进行语义分割，然后再计算loss

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
        loss_focal = self.focal_loss(sam_seg, gt_sam_inst_masks.unsqueeze(1)) * 20.0  # follow sam paper

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
        iou_thr = self.test_cfg.get("iou_thr", 0.7)  # 小图与大图的iou阈值

        pad_t = h_crop - h_stride
        pad_b = pad_t
        pad_l = w_crop - w_stride
        pad_r = pad_l
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b))
        inst_id = 1

        # gt的prompt也要相应位移
        if "gt_sam_prompt_points" in kwargs:
            assert len(kwargs["gt_sam_prompt_points"]) == 1
            kwargs["gt_sam_prompt_points"][0][:, :, :, 0] += pad_l
            kwargs["gt_sam_prompt_points"][0][:, :, :, 1] += pad_t

        batch_size, _, h_img, w_img = img.size()   #[bs, num_chan, h, w]
        assert batch_size == 1
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # grids是滑动窗口数量
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        inst_preds = img.new_zeros((h_img, w_img)).type(torch.int32)
        inst_score = {}
        inst_labels = {}
        inst_softmax = {}

        # 初始化用于存储整个图的 feat_embed 和 heat_preds
        # samseg_total = img.new_zeros(( h_img, w_img))
        # count_map = torch.zeros( h_img, w_img).to(img.device)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]   #从大图中通过滑动窗口裁剪crop size的小图，shape：[bs, num_chann, 256, 256] 
                # normalize
                # for channel in range(crop_img.shape[1]):
                #     img_c = crop_img[0, channel, ...]
                #     i99 = torch.quantile(img_c, 0.99)
                #     i1 = torch.quantile(img_c, 0.01)
                #     if i99 - i1 > 1e-3:
                #         norm_img_c = (img_c - i1) / (i99 - i1)
                #         crop_img[0, channel, ...] = norm_img_c
                #     else:
                #         crop_img[0, channel, ...] = 0

                # pred_scores_total: 列表的列表，元素为每个细胞核prompt的热图分数
                # sam_seg_total: 列表的列表，元素为每个实例的mask矩阵 [num_inst, 256, 256]
                # inst_types_total: 字典，每个实例的类型（语义分割）
                # softmax_result: 每个实例的类型的概率（语义分割）
                pred_scores_total, sam_seg_total, inst_types_total, softmax_result_total = self.encode_decode(
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

                # (self.num_classes-1)*[num_inst, ] prompter输出的概率
                pred_scores = pred_scores_total[0]
                # list长度：(self.num_classes-1) * [num_inst, 256, 256] SAM输出, before sigmoid
                # 这里的self.num_classes是2（前景和背景），不是细胞核类型数量
                sam_seg = sam_seg_total[0]
                inst_types = inst_types_total[0]
                softmax_results = softmax_result_total[0]  #[num_classes, num]

                # 过滤，nms
                # seg_masks: [num, 256, 256] 0/1矩阵
                # inst_types: [num, ] 目标类型，没有指定则为全0
                # pred_scores: [num, ] 目标得分
                inst_types = torch.cat(inst_types)
                pred_scores = torch.cat(pred_scores, dim=0)
                sam_seg = torch.cat(sam_seg, dim=0)  #[66, 256, 256]
                softmax_results = torch.cat(softmax_results, dim=1)

                seg_masks, inst_types, pred_scores, softmax_results = self.postprocess(
                    pred_scores, sam_seg, inst_types, softmax_results
                )
                if seg_masks is None:
                    continue
                # samseg_total, count_map = self.extract_mask(img, sam_seg, samseg_total, count_map,
                #                                 h_grids, w_grids,
                #                                 h_stride, w_stride,
                #                                 h_crop, w_crop)

                for idx in range(seg_masks.shape[0]):
                    seg_ = seg_masks[idx]
                    label_ = int(inst_types[idx])
                    score_ = float(pred_scores[idx])
                    softmax_ = softmax_results[:,idx].tolist()
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
                            inst_softmax[inst_id] = softmax_
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
                                > iou_thr
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
                                    inst_softmax[inst_id] = softmax_
                                    inst_id += 1
                            else:
                                inst_preds[y1:y2, x1:x2] = torch.where(
                                    focus_area > 0,
                                    focus_area,
                                    seg_.type(torch.int32) * (inst_id),
                                )
                                inst_score[inst_id] = score_
                                inst_labels[inst_id] = label_
                                inst_softmax[inst_id] = softmax_
                                inst_id += 1

        # embed_dict = self.extract_feat(img, img_meta)
        # feat_embed = embed_dict["feat_embed"]
        # heat_score = embed_dict["heat_score"]

        if pad_b == 0:
            pad_b = -inst_preds.shape[0]
        if pad_r == 0:
            pad_r = -inst_preds.shape[1]
        inst_preds = inst_preds[pad_t:-pad_b, pad_l:-pad_r]
        # feat_embed = feat_embed[:, pad_t:-pad_b, pad_l:-pad_r]
        # heat_score = heat_score[pad_t:-pad_b, pad_l:-pad_r]
        # samseg_mask = samseg_total[pad_t:-pad_b, pad_l:-pad_r]

        for ui in torch.unique(inst_preds):
            if ui == 0:
                continue
            if torch.sum(inst_preds == ui) < min_px:
                inst_preds[inst_preds == ui] = 0

        if rescale:
            w, h = img_meta[0]["ori_shape"][:2][::-1]
            inst_preds = mmcv.imresize(
                inst_preds.cpu().numpy(),
                (w, h),
                interpolation="nearest",
            )
            # resized_heat_score = F.interpolate(heat_score.unsqueeze(0).unsqueeze(0) , size=(w,h), mode='nearest')
            # resized_samseg = F.interpolate(samseg_mask.unsqueeze(0).unsqueeze(0) , size=(w,h), mode='bilinear', align_corners=False)
            # resized_heat_score = mmcv.imresize(
            #      heat_score.cpu().numpy(),
            #     (w, h),
            #     interpolation="nearest",
            # )

        # 创建字典
        # data_dict = {
        #     'heat_preds': resized_heat_score,
        #     'feat_embed': resized_feat_embed
        # }
        # torch.save(samseg_mask, "data_tmp/samseg.pth")

        return {
            "inst_preds": torch.from_numpy(inst_preds),  #掩码，sam的输出
            "inst_score": inst_score,     # 掩码置信度分数，只与prompt有关
            "inst_labels": inst_labels,
            "inst_softmax": inst_softmax,
        }

    def whole_inference(self, img, img_meta, rescale, **kwargs):
        """Inference with full image."""
        min_px = self.test_cfg.get("min_px", 16)  # 过滤小面积
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值
        iou_thr = self.test_cfg.get("iou_thr", 0.7)  # 小图与大图的iou阈值
        inst_id = 1

        batch_size, _, h_img, w_img = img.size()
        assert batch_size == 1
        num_classes = self.num_classes
        inst_preds = img.new_zeros((h_img, w_img)).type(torch.int32)
        inst_score = {}
        inst_labels = {}

        pred_scores_total, sam_seg_total, inst_types_total = self.encode_decode(
            img,
            img_meta,
            y1=0,
            x1=0,
            y2=h_img,
            x2=w_img,
            **kwargs,
        )

        if len(pred_scores_total) == 0:
            return {
                "inst_preds": inst_preds.cpu(),
                "inst_score": inst_score,
                "inst_labels": inst_labels,
            }

        # (self.num_classes-1)*[num, ] prompter输出的概率
        pred_scores = pred_scores_total[0]
        # (self.num_classes-1)*[num, 256, 256] SAM输出, before sigmoid
        sam_seg = sam_seg_total[0]
        inst_types = inst_types_total[0]
        if len(pred_scores) == 0:
            return {
                "inst_preds": inst_preds.cpu(),
                "inst_score": inst_score,
                "inst_labels": inst_labels,
            }

        # 过滤，nms
        # seg_masks: [num, 256, 256] 0/1矩阵
        # inst_types: [num, ] 目标类型，没有指定则为全0
        # pred_scores: [num, ] 目标得分
        inst_types = torch.cat(inst_types)
        pred_scores = torch.cat(pred_scores, dim=0)
        sam_seg = torch.cat(sam_seg, dim=0)

        seg_masks, inst_types, pred_scores = self.postprocess(
            pred_scores, sam_seg, inst_types
        )
        if seg_masks is None:
            return {
                "inst_preds": inst_preds.cpu(),
                "inst_score": inst_score,
                "inst_labels": inst_labels,
            }
        for idx in range(seg_masks.shape[0]):
            seg_ = seg_masks[idx]
            label_ = int(inst_types[idx])
            score_ = float(pred_scores[idx])
            inst_preds = torch.where(
                inst_preds > 0,
                inst_preds,
                seg_.type(torch.int32) * (inst_id),
            )
            inst_score[inst_id] = score_
            inst_labels[inst_id] = label_
            inst_id += 1

        for ui in torch.unique(inst_preds):
            if ui == 0:
                continue
            if torch.sum(inst_preds == ui) < min_px:
                inst_preds[inst_preds == ui] = 0
                del inst_score[int(ui)]
                del inst_labels[int(ui)]

        return {
            "inst_preds": inst_preds.cpu(),
            "inst_score": inst_score,
            "inst_labels": inst_labels,
        }

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
            output_dict = self.whole_inference(img, img_meta, rescale, **kwargs)
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

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        total_output_dict = []
        for i in range(len(imgs)):
            kwargs_single = {k: v[i : i + 1] for k, v in kwargs.items()}
            output_dict = self.inference(
                imgs[i], img_metas[i], rescale, **kwargs_single
            )
            total_output_dict.append(output_dict)

        total_pred_scores = []
        total_sam_seg = []
        for output_dict in total_output_dict:
            inst_preds = output_dict["inst_preds"]
            for i in torch.unique(inst_preds):
                if i == 0:
                    continue
                seg = inst_preds.new_full(inst_preds.shape, 0)
                seg[inst_preds == i] = 1
                score = output_dict["inst_score"][int(i)]
                label = output_dict["inst_labels"][int(i)]
                total_sam_seg.append(seg)
                total_pred_scores.append(score)

        total_pred_scores = torch.tensor(total_pred_scores)
        total_sam_seg = torch.stack(total_sam_seg, 0)
        seg_masks, inst_types, pred_scores = self.postprocess(
            total_pred_scores, total_sam_seg
        )
        inst_preds = seg_masks.new_full(seg_masks.shape[-2:], 0).type(torch.int32)
        inst_score = {}
        inst_labels = {}
        for i in range(seg_masks.shape[0]):
            seg_ = seg_masks[i]
            label = inst_types[i]
            score = pred_scores[i]
            inst_preds[seg_] = i + 1
            inst_score[i + 1] = float(score)
            inst_labels[i + 1] = float(label)
        return {
            "inst_preds": inst_preds.cpu().numpy(),
            "inst_score": inst_score,
            "inst_labels": inst_labels,
        }

    def extract_mask(self, img, sam_segmask, samseg_total, count_map,
                     h_grids,w_grids, 
                     h_stride, w_stride,
                     h_crop, w_crop
                     ):
        
        _, _, h_img, w_img = img.shape

        # 遍历滑动窗口
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # 累加 feat_embed 和 heat_preds
                samseg_total[y1:y2, x1:x2] += sam_segmask
                count_map[y1:y2, x1:x2] += 1

        return samseg_total, count_map


    def extract_feat(self, img, img_metas):
        """Placeholder for extract features from images."""
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size

        batch_size, _, h_img, w_img = img.size()   #[bs, num_chan, h, w]
        assert batch_size == 1
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # grids是滑动窗口数量
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
        dummy_crop = img[:, :, 0:h_crop, 0:w_crop]
        dummy_out = self.prompter.forward_test(dummy_crop, img_metas)   
        dummy_heat_preds = dummy_out["heat_preds"]
        dummy_heat_preds = F.interpolate(   #[num_gt, 256, 256]
                dummy_heat_preds,
                size=dummy_crop.shape[-2:],
                mode="nearest"
            )
        dummy_feat_embed = dummy_out["feat_embed"]
        dummy_feat_embed = F.interpolate(   #[num_gt, 256, 256]
                dummy_feat_embed,
                size=dummy_crop.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        _, num_feat_chann, H_feat, W_feat = dummy_feat_embed.shape

        # 计算特征图的整体尺寸
        scale_factor_h = H_feat / h_crop
        scale_factor_w = W_feat / w_crop
        total_feat_h = int(h_img * scale_factor_h)
        total_feat_w = int(w_img * scale_factor_w)

        # 初始化用于存储整个图的 feat_embed 和 heat_preds
        # heat_preds_total = img.new_zeros((h_img, w_img))
        heat_preds_total_max = img.new_full((h_img, w_img), float('-inf'))  # 初始化为负无穷
        heat_preds_total_min = img.new_full((h_img, w_img), float('inf'))   # 初始化为正无穷

        feat_embed_total = img.new_zeros(( num_feat_chann, h_img, w_img))
        count_map = torch.zeros(total_feat_h, total_feat_w).to(img.device)

        # 遍历滑动窗口
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2] 

                # 前向传播获取 feat_embed 和 heat_preds
                out = self.prompter.forward_test(crop_img, img_metas)
                heat_preds = out["heat_preds"]  # [1, num_classes, H_feat, W_feat]
                heat_preds = F.interpolate(   #[num_gt, 256, 256]
                    heat_preds,
                    size=crop_img.shape[-2:],
                    mode="nearest"
                ).squeeze()
                # feat_embed = out["feat_embed"]  # [1, num_classes, H_feat, W_feat]
                # feat_embed = F.interpolate(   #[num_gt, 256, 256]
                #     feat_embed,
                #     size=crop_img.shape[-2:],
                #     mode="bilinear",
                #     align_corners=False,
                # ).squeeze(0)

                # 更新 heat_preds_total的最大值和最小值
                heat_preds_total_max[y1:y2, x1:x2] = torch.maximum( heat_preds_total_max[y1:y2, x1:x2], heat_preds)
                heat_preds_total_min[y1:y2, x1:x2] = torch.minimum( heat_preds_total_min[y1:y2, x1:x2], heat_preds)

                # 累加 feat_embed
                # feat_embed_total[:, y1:y2, x1:x2] += feat_embed
                # count_map[y1:y2, x1:x2] += 1

        # 处理重叠区域，通过平均化
        # heat_preds_total /= count_map
        # feat_embed_total /= count_map.unsqueeze(0)

        return {"heat_score_max": heat_preds_total_max, "heat_score_min": heat_preds_total_min}


    def encode_decode(self, img, img_metas, **kwargs):
        y1 = kwargs["y1"]
        x1 = kwargs["x1"]
        y2 = kwargs["y2"]
        x2 = kwargs["x2"]
        center_xy = kwargs.get("center_xy", None)
        # list of tensor[1, n, 1+neg, 2]
        gt_sam_prompt_points = kwargs.get("gt_sam_prompt_points", None)
        # list of tensor[1, n, 1+neg]
        gt_sam_prompt_labels = kwargs.get("gt_sam_prompt_labels", None)

        gt_semantic_seg = kwargs.get("gt_semantic_seg", None)

        # 使用gt测试上限
        if gt_sam_prompt_points is not None:
            # [n, 1, 1+neg, 2]
            gt_sam_prompt_points = (
                torch.cat(gt_sam_prompt_points, dim=0).squeeze(0).unsqueeze(1)
            )
            # [n, 1, 1+neg]
            gt_sam_prompt_labels = (
                torch.cat(gt_sam_prompt_labels, dim=0).squeeze(0).unsqueeze(1)
            )
            indexes = torch.where(
                (gt_sam_prompt_points[..., 0, 0] > x1)
                & (gt_sam_prompt_points[..., 0, 0] < x2)
                & (gt_sam_prompt_points[..., 0, 1] > y1)
                & (gt_sam_prompt_points[..., 0, 1] < y2)
            )
            gt_sam_prompt_points = gt_sam_prompt_points[indexes]  # [n, 1+neg ,2]
            gt_sam_prompt_labels = gt_sam_prompt_labels[indexes]  # [n, 1+neg]
            gt_sam_prompt_points[..., 0] -= x1
            gt_sam_prompt_points[..., 1] -= y1
            cell_num = gt_sam_prompt_points.shape[0]
            cls_preds = torch.cat(gt_semantic_seg, dim=0)  # [bs, 256, 256]
            one_hot_preds = torch.nn.functional.one_hot(cls_preds.to(torch.int64), num_classes=self.num_classes)
            softmax_result = one_hot_preds.permute(0,3,1,2)

        else:
            score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值
            out_dict = self.prompter.forward_test(img, img_metas)   
            heat_preds = out_dict["heat_preds"]  # [bs, 1(num_classes-1), 64, 64], 64是因为下采样
            heat_preds = heat_preds.sigmoid()    # [bs, 1(num_classes-1), 64, 64]
            seg_preds = out_dict["seg_preds"]    # 这是分类的概率 [bs, num_classes, 64, 64]
            softmax_result = seg_preds.softmax(1)   # [bs, num_classes, 64, 64]
            cls_preds = seg_preds.max(dim=1)[1]  # 取最大数值的索引
            if self.num_classes==2:
                cls_preds = torch.ones_like(cls_preds, dtype=torch.long, device=cls_preds.device)

        sam_seg_total = []
        pred_scores_total = []
        inst_types_total = []
        softmax_result_total = []
        for batch_idx in range(img.shape[0]):
            sam_seg_c = []
            pred_scores_c = []
            inst_types = []
            softmax_results = []
            # for c in range(self.num_classes - 1):  # self.num_classes包括背景
            if gt_sam_prompt_points is None:
                heat_preds_single = heat_preds[batch_idx, 0]  #选择灰度图像通道
                # 选取prompt的坐标
                indices = torch.where(heat_preds_single > score_thr)   # 细胞核中心坐标位置（y，x）
                pred_scores = heat_preds_single[indices]   #细胞核中心的分数

                # 将细胞核中心坐标作为sam的提示输入
                pred_points_sam = list(
                    zip(indices[1].tolist(), indices[0].tolist())
                )  #  [num, 2], [(x, y), ] x和y反过来
                if len(pred_points_sam) > 0:
                    pred_points_sam = torch.tensor(
                        pred_points_sam, device=img.device, dtype=torch.float32
                    )
                    pred_points_sam = pred_points_sam * self.heat_stride  # heat_preds为原图的1/4
                    pred_points_sam = pred_points_sam.unsqueeze(1)  # [num_gt, 1, 2]，每个实例的prompt点坐标
                    label_sam = torch.ones(
                        (pred_points_sam.shape[0], 1),
                        dtype=torch.int32,
                        device=img.device,
                    )  # [num_gt, 1]

                    if self.num_neg_prompt > 0:
                        from mmseg.datasets.pipelines.transforms import GetHeatMap

                        (
                            pred_points_sam,
                            label_sam,
                        ) = GetHeatMap.add_k_nearest_neg_prompt(
                            pred_points_sam,
                            torch.arange(len(pred_points_sam)),
                            pred_points_sam,
                            k=self.num_neg_prompt,
                            min_dis=16,
                        )
                    else:
                        # pad
                        pred_points_sam = torch.cat(
                            [
                                pred_points_sam,
                                torch.zeros_like(
                                    pred_points_sam,
                                    dtype=pred_points_sam.dtype,
                                    device=pred_points_sam.device,
                                ),
                            ],
                            dim=1,
                        )  # shape: [num_gt, 2, 2]，相当于在pos_prompt之后又增加一行neg_prompt，是[0,0]
                        label_sam = torch.cat(
                            [
                                label_sam,
                                -torch.ones_like(
                                    label_sam,
                                    dtype=label_sam.dtype,
                                    device=label_sam.device,
                                ),
                            ],
                            dim=1,
                        )

                    sam_output = self.sam(
                        img[batch_idx].unsqueeze(0),
                        pred_points_sam.unsqueeze(1),
                        label_sam.unsqueeze(1),
                        multimask_output=False,
                        cell_num=len(pred_points_sam),
                    )
                else:
                    continue
            else:
                if gt_sam_prompt_points.shape[0] > 0:
                    # 使用gt的prompts, 测试上限
                    sam_output = self.sam(
                        img[batch_idx].unsqueeze(0),
                        gt_sam_prompt_points.unsqueeze(1),
                        gt_sam_prompt_labels.unsqueeze(1),
                        multimask_output=False,
                        cell_num=cell_num,
                    )
                    pred_scores = torch.ones(
                        (sam_output[1].shape[0],), device=img.device
                    ).type(torch.float32)
                    indices = (
                        gt_sam_prompt_points[:, 0, 1].long(),
                        gt_sam_prompt_points[:, 0, 0].long(),
                    )
                else:
                    continue
            sam_seg = sam_output[1].squeeze(1)  # [num_gt, 1, 64, 64]，每个实例的mask
            # resize to gt_inst_mask size
            sam_seg = F.interpolate(   #[num_gt, 256, 256]
                sam_seg,
                size=img[batch_idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            sam_seg_c.append(sam_seg)
            pred_scores_c.append(pred_scores) 
            inst_types.append(cls_preds[batch_idx][indices] - 1)  # 后续计算分类从0开始
            softmax_results.append(softmax_result[batch_idx,:,indices[0],indices[1]])   #size[num_classes,num]
            
            sam_seg_total.append(sam_seg_c)
            pred_scores_total.append(pred_scores_c)
            inst_types_total.append(inst_types)
            softmax_result_total.append(softmax_results)

        return pred_scores_total, sam_seg_total, inst_types_total, softmax_result_total

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

    def postprocess(self, pred_scores, sam_seg, inst_types, softmax_result):
        nms_pre = self.test_cfg.get("nms_pre", 500)  # nms前的目标数
        max_per_img = self.test_cfg.get("max_per_img", 200)  # nms后的目标数
        update_thr = self.test_cfg.get("update_thr", 0.2)  # nms后score阈值

        seg_masks = sam_seg.sigmoid() > 0.5  # [num, 256, 256] 0/1 mask
        sum_masks = seg_masks.sum((1, 2)).float()  # [num_gt, ] 每个mask的面积
        # [num, ] 每个mask的平均概率
        seg_scores = (sam_seg.sigmoid() * seg_masks).sum((1, 2)) / (sum_masks + 1e-6)
        pred_scores *= seg_scores  # prompter输出的概率*每个mask的平均概率
        # pred_scores = seg_scores

        # min area filter.
        keep = sum_masks > 4
        if keep.sum() == 0:
            return [None] * 4
        
        seg_masks = seg_masks[keep, ...]  # [num, 256, 256] 0/1 mask
        sam_seg = sam_seg[keep, ...]  # [num, 256, 256] SAM输出, before sigmoid
        sum_masks = sum_masks[keep]  # [num_gt, ] 每个mask的面积
        pred_scores = pred_scores[keep]  # prompter输出的概率*每个mask的平均概率
        inst_types = inst_types[keep]
        softmax_result = softmax_result[:,keep]   # [num_classes, num_gt]

        # nms
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > nms_pre:
            sort_inds = sort_inds[:nms_pre]

        seg_masks = seg_masks[sort_inds, :, :]  # [num, 256, 256]
        sam_seg = sam_seg[sort_inds, :, :]  # [num, 256, 256]
        sum_masks = sum_masks[sort_inds]  # [num, ]
        pred_scores = pred_scores[sort_inds]  # [num, ]
        inst_types = inst_types[sort_inds]
        softmax_result = softmax_result[:,sort_inds]  # [num_classes, num]

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
            return [None] * 4
        sam_seg = sam_seg[keep, :, :]
        pred_scores = pred_scores[keep]
        inst_types = inst_types[keep]
        softmax_result = softmax_result[:,keep]

        # sort and keep top_k
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > max_per_img:
            sort_inds = sort_inds[:max_per_img]
        sam_seg = sam_seg[sort_inds, :, :]
        pred_scores = pred_scores[sort_inds]
        inst_types = inst_types[sort_inds]
        softmax_result = softmax_result[:,sort_inds]

        seg_masks = sam_seg.sigmoid() > 0.5

        return seg_masks, inst_types, pred_scores, softmax_result
