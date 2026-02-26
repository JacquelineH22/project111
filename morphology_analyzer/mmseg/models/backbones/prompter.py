import warnings
from collections import OrderedDict
from copy import deepcopy

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

from mmseg.core import add_prefix
from ...utils import get_root_logger
from ..utils import HungarianMatcher_Crowd
from ..builder import BACKBONES
from .. import builder


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = anchor_points.reshape((1, A, 2)) + shifts.reshape(
        (1, K, 2)
    ).transpose((1, 0, 2))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchor_points * 2, kernel_size=3, padding=1
        )

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchor_points=4,
        num_classes=80,
        prior=0.01,
        feature_size=256,
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(
            batch_size, width, height, self.num_anchor_points, self.num_classes
        )

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels  # [3,]

        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]

        self.row = row  # 2
        self.line = line  # 2

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [
            (image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels
        ]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(
                image_shapes[idx], self.strides[idx], anchor_points
            )
            all_anchor_points = np.append(
                all_anchor_points, shifted_anchor_points, axis=0
            )

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(image.device)


@BACKBONES.register_module()
class Prompter(BaseModule):
    # https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
    # P2PNet
    def __init__(
        self,
        backbone,
        neck,
        regression,
        matcher,
        row=2,
        line=2,
        num_classes=2,
        eos_coef=0.5,
        threshold=0.5,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = builder.build_backbone(backbone)
        self.num_classes = num_classes
        self.neck = builder.build_neck(neck)
        self.regression = RegressionModel(
            num_features_in=regression.num_features_in,
            num_anchor_points=row * line,
        )
        self.classification = ClassificationModel(
            num_features_in=regression.num_features_in,
            num_classes=self.num_classes,
            num_anchor_points=row * line,
        )
        self.anchor_points = AnchorPoints(
            pyramid_levels=[3],
            row=row,
            line=line,
        )
        self.matcher = HungarianMatcher_Crowd(
            cost_class=matcher.set_cost_class, cost_point=matcher.set_cost_point
        )

        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.threshold = threshold

    def forward_test(self, img, img_metas):
        features = self.backbone(img)
        features = self.neck(features)  # 4, 8, 16, 32

        batch_size = img.shape[0]
        regression = self.regression(features[1]) * 100  # 8xfeature, pyramid_levels=[3]
        classification = self.classification(features[1])
        anchor_points = self.anchor_points(img).repeat(batch_size, 1, 1)

        output_coord = regression + anchor_points  # [bs, num, 2]
        output_class = classification  # [bs, num, num_classes]
        out = {"pred_logits": output_class, "pred_points": output_coord}

        return out

    def forward_train(self, img, img_metas, gt_points):
        features = self.backbone(img)
        features = self.neck(features)  # 4, 8, 16, 32

        batch_size = img.shape[0]
        regression = self.regression(features[1]) * 100  # 8xfeature, pyramid_levels=[3]
        classification = self.classification(features[1])
        anchor_points = self.anchor_points(img).repeat(batch_size, 1, 1)

        output_coord = regression + anchor_points  # [bs, num, 2]
        output_class = classification  # [bs, num, num_classes]
        out = {"pred_logits": output_class, "pred_points": output_coord}

        targets = []
        for idx in range(len(gt_points)):
            targets.append(
                {
                    "point": gt_points[idx],
                    "labels": torch.ones(
                        len(gt_points[idx]), dtype=torch.int64, device=img.device
                    ),
                }
            )

        indices1 = self.matcher(out, targets)
        loss = self.losses(out, targets, indices1)
        loss = add_prefix(loss, "prompter")

        return loss, out, indices1

    def losses(self, out, target, indices):
        loss = dict()
        loss_labels = self.loss_labels(out, target, indices)
        loss_points = self.loss_points(out, target, indices)
        loss["loss_labels"] = loss_labels
        loss["loss_points"] = loss_points
        return loss

    def loss_labels(self, out, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = out["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )  # [bs, num_query]
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )

        return loss_ce

    def loss_points(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.mse_loss(src_points, target_points, reduction="none")

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=src_points.device
        )
        loss_bbox = loss_bbox.sum() / num_points

        return loss_bbox

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
