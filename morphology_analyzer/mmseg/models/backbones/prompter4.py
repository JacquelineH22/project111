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
from mmcv.runner import force_fp32, load_checkpoint

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import BACKBONES
from .. import builder
from ..decode_heads.pointnu_head import build_layer, JPFM, BinaryDiceLoss, CoordConv2d
from mmseg.utils import get_root_logger


@BACKBONES.register_module()
class Prompter4(BaseModule):
    # 使用vit-pose生成heatmap
    def __init__(
        self,
        backbone,
        keypoint_head,
        alpha=2,
        beta=4,
        init_cfg=None,
        neck=None,
        num_classes=1,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None
        self.keypoint_head = builder.build_head(keypoint_head)

        self.ins_loss = BinaryDiceLoss()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward_test(self, img, img_metas):
        return self.forward(img, img_metas)

    def forward_train(self, img, img_metas, gt_heat_map):
        out = self.forward(img, img_metas)

        heat_preds = out["heat_preds"]

        loss = dict()
        loss["loss_heat"] = self.local_focal(heat_preds.sigmoid(), gt_heat_map)

        loss = add_prefix(loss, "prompter")

        return {"loss": loss, "out": out}

    def forward(self, img, img_metas):
        x = self.backbone(img)  # bs, 1024, 16, 16 vit16倍下采样

        x = self.keypoint_head(x)  # bs, num_classes, 64, 64

        out = {"heat_preds": x}
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

    def init_weights(self) -> None:
        if not self._is_init and self.init_cfg is not None:
            logger = get_root_logger()
            load_checkpoint(
                self,
                self.init_cfg.get("checkpoint"),
                map_location="cpu",
                logger=logger,
                revise_keys=[(r"^module\.", "")],
            )
            self._is_init = True
