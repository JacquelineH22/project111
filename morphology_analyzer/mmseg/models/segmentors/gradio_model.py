import math
from collections import Counter

import torch
import torch.nn.functional as F

from ..builder import SEGMENTORS
from .prompt_sam import PromptSam
from .finer_prompt_sam import FinerPromptSam


@SEGMENTORS.register_module()
class GradioModel(PromptSam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_embeddings = []
        self.crop_centers = []  # xy
        self.lefttop = []  # xy

    def predict(self, img, prompt_points=None, prompt_labels=None):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        min_px = self.test_cfg.get("min_px", 16)  # 过滤小面积
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值
        iou_thr = self.test_cfg.get("iou_thr", 0.7)  # 小图与大图的iou阈值

        pad_t = h_crop - h_stride
        pad_b = pad_t
        pad_l = w_crop - w_stride
        pad_r = pad_l
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        inst_id = 1

        if prompt_points is not None:
            prompt_points = torch.tensor(prompt_points)  # [1, 2] 或者 [2, 2]
            prompt_labels = torch.tensor(prompt_labels)  #
            prompt_points[:, 0] += pad_l
            prompt_points[:, 1] += pad_t

        batch_size, _, h_img, w_img = img.size()
        assert batch_size == 1

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        inst_preds = img.new_zeros((h_img, w_img), dtype=torch.int32)
        cls_preds = img.new_zeros((self.num_classes, h_img, w_img), dtype=torch.float32)
        counts = img.new_zeros((h_img, w_img)).type(
            torch.float32
        )  # 用于平均type_preds_unsig
        inst_score = {}
        inst_labels = {}

        if len(self.image_embeddings) > 0:
            distance = torch.square(
                self.crop_centers - prompt_points[0].unsqueeze(0)
            ).sum(1)
            index = distance.argmin()
            image_embeddings = self.image_embeddings[index]
            left, top = self.lefttop[index]
            prompt_points[:, 0] -= left
            prompt_points[:, 1] -= top
            sam_output = self.sam(
                pixel_values=None,
                input_points=prompt_points[None, None, :].cuda(),
                input_labels=prompt_labels[None, None, :].cuda(),
                multimask_output=False,
                cell_num=1,
                image_embeddings=image_embeddings.cuda(),
            )
            sam_seg = sam_output[1].squeeze(1)  # [num_gt, 1, 64, 64]
            # resize to gt_inst_mask size
            sam_seg = (
                F.interpolate(
                    sam_seg,
                    size=(h_crop, w_crop),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(0)
            )  # [256, 256]
            sam_mask = (sam_seg.sigmoid() > 0.5).type(torch.int32).cpu()
            inst_preds[top : top + 256, left : left + 256] = sam_mask
            inst_score = {}
            inst_labels = {}
            cls_preds = img.new_zeros(
                (h_img, w_img), dtype=torch.float32
            )

        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    self.crop_centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
                    self.lefttop.append([x1, y1])
                    crop_img = img[:, :, y1:y2, x1:x2]
                    pred_scores, sam_seg, inst_types, cls_preds_crop, softmax_result = (
                        self.forward_prompt_sam(crop_img)
                    )
                    counts[y1:y2, x1:x2] += 1
                    cls_preds[:, y1:y2, x1:x2] += cls_preds_crop
                    if sam_seg is None:
                        continue
                    
                    # softmax_result = softmax_result[0]
                    seg_masks, inst_types, pred_scores, _ = self.postprocess(
                        pred_scores, sam_seg, inst_types, softmax_result
                    )

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
                                        torch.logical_and(
                                            focus_area == compared_num, seg_
                                        )
                                    )
                                    / torch.sum(
                                        torch.logical_or(
                                            focus_area == compared_num, seg_
                                        )
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

            self.crop_centers = torch.tensor(self.crop_centers)
            cls_preds = cls_preds / counts.unsqueeze(0)
            cls_preds = cls_preds.max(dim=0)[1] - 1  # 从0开始
        
        embed_dict = self.extract_feat_app(img)
        heat_score = embed_dict["heat_score_min"]

        if pad_b == 0:
            pad_b = -inst_preds.shape[0]
        if pad_r == 0:
            pad_r = -inst_preds.shape[1]
        inst_preds = inst_preds[pad_t:-pad_b, pad_l:-pad_r]
        cls_preds = cls_preds[pad_t:-pad_b, pad_l:-pad_r]
        heat_score = heat_score[pad_t:-pad_b, pad_l:-pad_r]
        heat_score_sig = heat_score.sigmoid().detach().cpu().numpy()

        for ui in torch.unique(inst_preds):
            if ui == 0:
                continue
            if torch.sum(inst_preds == ui) < min_px:
                inst_preds[inst_preds == ui] = 0

        # === 新增：根据 clicked_point 返回 heat_score_min ===
        # clicked_min_val = None
        # if clicked_point is not None and hasattr(self, "heat_preds_total_min"):
        #     x_clicked, y_clicked = clicked_point  # (x, y)
        #     # 注意：如果 predict() 中还有 padding，这里可能需要做一次坐标映射
        #     # 这里假设 clicked_point 已经是去掉 padding 后的坐标
        #     if 0 <= x_clicked < self.heat_preds_total_min.shape[1] and 0 <= y_clicked < self.heat_preds_total_min.shape[0]:
        #         val = self.heat_preds_total_min[y_clicked+pad_t, x_clicked+pad_b]
        #         clicked_min_val = float(val.item())

        return {
            "inst_preds": inst_preds.cpu(),
            "inst_score": inst_score,
            "inst_labels": inst_labels,
            "cls_preds": cls_preds,
            "clicked_min_val":heat_score_sig
        }

    def forward_prompt_sam(self, crop_img):
        score_thr = self.test_cfg.get("score_thr", 0.4)  # nms前score阈值
        out_dict = self.prompter.forward_test(crop_img, img_metas=None)
        heat_preds = out_dict["heat_preds"]  # [bs, 1(num_classes-1), 64, 64]
        heat_preds = heat_preds.sigmoid()
        seg_preds = out_dict["seg_preds"]  #[bs, num_classes, 64, 64]
        softmax_result = torch.softmax(seg_preds, dim=1)  #[bs, num_classes, 64, 64]
        cls_preds = seg_preds.max(dim=1)[1]

        # if self.num_classes == 2:
        #     cls_preds = torch.ones_like(
        #         cls_preds, dtype=torch.long, device=cls_preds.device
        #     )

        indices = torch.where(heat_preds > score_thr)
        pred_scores = heat_preds[indices]
        pred_points_sam = list(
            zip(indices[-1].tolist(), indices[-2].tolist())
        )  #  [num, 2], [(x, y), ]
        if len(pred_points_sam) == 0:
            return [None] * 4
        if len(pred_points_sam) > 0:
            pred_points_sam = torch.tensor(
                pred_points_sam, device=crop_img.device, dtype=torch.float32
            )
            pred_points_sam = (
                pred_points_sam * self.heat_stride
            )  # heat_preds为原图的1/4
            pred_points_sam = pred_points_sam.unsqueeze(1)  # [num, 1, 2]
            label_sam = torch.ones(
                (pred_points_sam.shape[0], 1),
                dtype=torch.int32,
                device=crop_img.device,
            )
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
            )
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
                crop_img,
                pred_points_sam.unsqueeze(1),
                label_sam.unsqueeze(1),
                multimask_output=False,
                cell_num=1,
            )
            sam_seg = sam_output[1].squeeze(1)  # [num_gt, 1, 64, 64]
            # resize to gt_inst_mask size
            sam_seg = F.interpolate(
                sam_seg,
                size=crop_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            inst_types = cls_preds[indices[0], indices[2], indices[3]] - 1
            softmax_result = softmax_result[0,:,indices[0],indices[1]]  #[num_classes, num_gt]

            self.image_embeddings.append(sam_output["vision_hidden_states"].cpu())
            seg_preds = F.interpolate(
                seg_preds,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )

            return pred_scores, sam_seg, inst_types, seg_preds[0], softmax_result


    def extract_feat_app(self, img):
            """Placeholder for extract features from images."""
            h_stride, w_stride = self.test_cfg.stride
            h_crop, w_crop = self.test_cfg.crop_size

            batch_size, _, h_img, w_img = img.size()   #[bs, num_chan, h, w]
            assert batch_size == 1
            num_classes = self.num_classes
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # grids是滑动窗口数量
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            
            dummy_crop = img[:, :, 0:h_crop, 0:w_crop]
            dummy_out = self.prompter.forward_test(dummy_crop, img_metas=None) 
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

            # feat_embed_total = img.new_zeros(( num_feat_chann, h_img, w_img))
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
                    out = self.prompter.forward_test(crop_img, img_metas=None) 
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

            # ========= 将结果存储到 self 中，便于在 predict 中访问 =========
            self.heat_preds_total_max = heat_preds_total_max
            self.heat_preds_total_min = heat_preds_total_min
            # self.feat_embed_total = feat_embed_total

            return {"heat_score_max": heat_preds_total_max, "heat_score_min": heat_preds_total_min}
