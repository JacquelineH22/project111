import warnings
from collections import OrderedDict
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_, trunc_normal_init
from mmcv.runner import BaseModule, CheckpointLoader, ModuleList, load_state_dict
from mmcv.utils import to_2tuple
from mmcv.runner import force_fp32

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import BACKBONES, build_loss
from .. import builder
from ..decode_heads.pointnu_head import build_layer, JPFM, BinaryDiceLoss, CoordConv2d


class _PointNuNetHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=256 * 4,
        seg_feat_channels=256,
        stacked_convs=7,
        ins_out_channels=256,
        kernel_size=1,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super(_PointNuNetHead, self).__init__()
        self.num_classes = num_classes
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.seg_feat_channels = seg_feat_channels
        self.seg_out_channels = ins_out_channels
        self.ins_out_channels = ins_out_channels
        self.kernel_out_channels = self.ins_out_channels * kernel_size * kernel_size
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.num_classes = num_classes

        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        # self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            conv = CoordConv2d if i == 0 else nn.Conv2d

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                    build_norm_layer(self.norm_cfg, self.seg_feat_channels)[1],
                    nn.ReLU(True),
                )
            )

        self.head_cate = nn.Conv2d(
            self.seg_feat_channels, 1, 3, padding=1
        )  # 热图1通道，分类任务由分割头负责

        self.mask_convs.append(
            nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                build_norm_layer(self.norm_cfg, self.seg_feat_channels)[1],
                nn.ReLU(True),
                nn.Conv2d(
                    self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                build_norm_layer(self.norm_cfg, self.seg_feat_channels)[1],
                nn.ReLU(True),
            )
        )

    def init_weight(self):
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.head_cate.weight, std=0.01)
        torch.nn.init.constant_(self.head_cate.bias, bias_init)

    def forward(self, feats):
        # cate branch
        cate_feat = feats
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.head_cate(cate_feat)  # [bs, 1, 64, 64]
        # return feature_pred, kernel_pred, cate_pred
        return cate_pred


@BACKBONES.register_module()
class Prompter3(BaseModule):
    # prompt by heatmap
    def __init__(
        self,
        backbone,
        num_classes,
        in_index,
        in_channels,
        align_corners,
        decoder_params,
        pointnu_head_cfg,
        init_cfg=None,
        neck=None,
        alpha=2,
        beta=4,
        with_mse=False,
        loss_decode=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = builder.build_backbone(backbone)
        self.num_classes = num_classes
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        self.in_index = in_index
        self.in_channels = in_channels
        self.align_corners = align_corners
        self.alpha = alpha
        self.beta = beta
        self.with_mse = with_mse

        embed_dims = decoder_params["embed_dims"]
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params["embed_cfg"]
        embed_neck_cfg = decoder_params["embed_neck_cfg"]
        if embed_neck_cfg == "same_as_embed_cfg":
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params["fusion_cfg"]
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and "aspp" in cfg["type"]:
                cfg["align_corners"] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(
            self.in_index, self.in_channels, embed_dims
        ):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg
                )
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg
                )
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        # self.fuse_layer = build_layer(
        #     sum(embed_dims), self.channels, **fusion_cfg)
        num_inputs = len(self.in_channels)
        # self.channels = sum(embed_dims)

        self.jpfm_1 = JPFM(in_channel=sum(embed_dims))
        self.heads = _PointNuNetHead(
            num_classes=self.num_classes,
            in_channels=self.jpfm_1.out_channel,
            seg_feat_channels=pointnu_head_cfg["seg_feat_channels"],
            stacked_convs=pointnu_head_cfg["stacked_convs"],
            ins_out_channels=pointnu_head_cfg["ins_out_channels"],
            kernel_size=pointnu_head_cfg["kernel_size"],
            norm_cfg=pointnu_head_cfg["norm_cfg"],
        )

        self.ins_loss = BinaryDiceLoss()

    def forward_test(self, img, img_metas):
        return self.forward(img, img_metas)

    def forward_train(self, img, img_metas, gt_heat_map, **kwargs):
        out = self.forward(img, img_metas)
        gt_semantic_seg = kwargs["gt_semantic_seg"]

        heat_preds = out["heat_preds"]

        loss = dict()
        loss["loss_heat"] = self.local_focal(heat_preds.sigmoid(), gt_heat_map)

        if self.with_mse:
            loss["loss_heat_mse"] = F.mse_loss(
                heat_preds.sigmoid(), gt_heat_map.float()
            )

        loss = add_prefix(loss, "prompter")

        return {"loss": loss, "out": out}

    def forward(self, img, img_metas):
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)  # 4, 8, 16, 32x features

        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = (
                    _c[i]
                    .permute(0, 2, 1)
                    .contiguous()
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                )
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )

        x = torch.cat(list(_c.values()), dim=1)
        f1 = self.jpfm_1(x)  # [bs, 1024, 64, 64]
        heat_preds = self.heads(f1)

        out = {"heat_preds": heat_preds}
        return out

    def local_focal(self, pred, gt):
        """
        focal loss copied from CenterNet, modified version focal loss
        change log: numeric stable version implementation
        """
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        neg_weights = torch.pow(1 - gt[neg_inds], self.beta)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, self.alpha)
        neg_loss = (
            torch.log(1 - neg_pred) * torch.pow(neg_pred, self.alpha) * neg_weights
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # print(num_pos)

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss
