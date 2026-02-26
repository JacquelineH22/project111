# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .sep_aspp_head import DepthwiseSeparableASPPModule
from ..losses import accuracy


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction="mean", **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = (
            1  # suggest set a large number when target area is large,like '10|100'
        )
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if "batch_loss" in kwargs.keys():
            self.batch_dice = kwargs["batch_loss"]

    def forward(self, output, target, use_sigmoid=True):
        assert (
            output.shape[0] == target.shape[0]
        ), f"output & target batch size don't match {output.shape[0]} {target.shape[0]} "
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class CoordConv2d(nn.Conv2d):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CoordConv2d, self).__init__(
            in_chan + 2,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        batchsize, H, W = x.size(0), x.size(2), x.size(3)
        h_range = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        w_range = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        h_chan, w_chan = torch.meshgrid(h_range, w_range)
        h_chan = h_chan.expand([batchsize, 1, -1, -1])
        w_chan = w_chan.expand([batchsize, 1, -1, -1])

        feat = torch.cat([h_chan, w_chan, x], dim=1)

        return F.conv2d(
            feat,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _PointNuNetHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=256 * 4,
        seg_feat_channels=256,
        stacked_convs=7,
        ins_out_channels=256,
        kernel_size=1,
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

        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            conv = CoordConv2d if i == 0 else nn.Conv2d
            self.kernel_convs.append(
                nn.Sequential(
                    conv(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.seg_feat_channels),
                    nn.ReLU(True),
                )
            )

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.seg_feat_channels),
                    nn.ReLU(True),
                )
            )

        self.head_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 1, padding=0
        )
        self.head_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1
        )

        self.mask_convs.append(
            nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
            )
        )

        self.mask_convs.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    4,
                    2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
            )
        )

        self.mask_convs.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    4,
                    2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False
                ),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
            )
        )

        self.head_mask = nn.Sequential(
            nn.Conv2d(
                self.seg_feat_channels, self.seg_out_channels, 1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.seg_out_channels),
            nn.ReLU(True),
        )

    def init_weight(self):
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.head_cate.weight, std=0.01)
        torch.nn.init.constant_(self.head_cate.bias, bias_init)

    def forward(self, feats, f2, f3):
        # feature branch
        mask_feat = feats
        for i, mask_layer in enumerate(self.mask_convs):
            mask_feat = mask_layer(mask_feat)  # [bs, 256, 256, 256]
        feature_pred = self.head_mask(mask_feat)  # [bs, 256, 256, 256]

        # kernel branch
        kernel_feat = f2
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.head_kernel(kernel_feat)  # [bs, 256, 64, 64]

        # cate branch
        cate_feat = f3
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.head_cate(cate_feat)  # [bs, 1, 64, 64]
        return feature_pred, kernel_pred, cate_pred


class JPFM(nn.Module):
    def __init__(self, in_channel, width=256):
        super(JPFM, self).__init__()

        self.out_channel = width * 4
        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
        )
        self.dilation4 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
        )

    def forward(self, feat):
        feat = torch.cat(
            [
                self.dilation1(feat),
                self.dilation2(feat),
                self.dilation3(feat),
                self.dilation4(feat),
            ],
            dim=1,
        )
        return feat


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class ASPPWrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        sep,
        dilations,
        pool,
        norm_cfg,
        act_cfg,
        align_corners,
        context_cfg=None,
    ):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
                ),
            )
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels, **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg,
        )
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == "id":
        return nn.Identity()
    elif type == "mlp":
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == "sep_conv":
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs["kernel_size"] // 2,
            **kwargs,
        )
    elif type == "conv":
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs["kernel_size"] // 2,
            **kwargs,
        )
    elif type == "aspp":
        return ASPPWrapper(in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == "rawconv_and_aspp":
        kernel_size = kwargs.pop("kernel_size")
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            ASPPWrapper(in_channels=out_channels, channels=out_channels, **kwargs),
        )
    elif type == "isa":
        return ISALayer(in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class PointNuHead(BaseDecodeHead):
    def __init__(self, pointnu_head_cfg, **kwargs):
        super(PointNuHead, self).__init__(input_transform="multiple_select", **kwargs)
        self.pointnu_head_cfg = pointnu_head_cfg
        assert not self.align_corners
        decoder_params = kwargs["decoder_params"]
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

        self.jpfm_1 = JPFM(in_channel=self.channels * num_inputs)
        self.jpfm_2 = JPFM(in_channel=self.channels * num_inputs)
        self.jpfm_3 = JPFM(in_channel=self.channels * num_inputs)
        self.heads = _PointNuNetHead(
            num_classes=self.num_classes,
            in_channels=self.jpfm_1.out_channel,
            seg_feat_channels=pointnu_head_cfg["seg_feat_channels"],
            stacked_convs=pointnu_head_cfg["stacked_convs"],
            ins_out_channels=pointnu_head_cfg["ins_out_channels"],
            kernel_size=pointnu_head_cfg["kernel_size"],
        )

        self.ins_loss = BinaryDiceLoss()

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

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
        f2 = self.jpfm_2(x)  # [bs, 1024, 64, 64]
        f3 = self.jpfm_3(x)  # [bs, 1024, 64, 64]
        feature_preds, kernel_preds, heat_preds = self.heads(f1, f2, f3)
        # [bs, 256, 256, 256]
        # [bs, 256, 64, 64]
        # [bs, num_classes, 64, 64]
        return feature_preds, kernel_preds, heat_preds

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        train_cfg,
        seg_weight,
        gt_heat_map,
        gt_inst_mask,
        gt_is_center,
    ):
        feature_preds, kernel_preds, heat_preds = self.forward(inputs)

        loss_inst = []
        loss_heat = []
        N, _, h, w = feature_preds.shape
        for batch_idx in range(N):
            feature_pred_single = feature_preds[batch_idx]  # [256, 256, 256]
            kernel_pred_single = kernel_preds[batch_idx]  # [256, 64, 64]
            heat_pred_single = heat_preds[batch_idx]  # [num_classes, 64, 64]
            gt_heat_map_single = gt_heat_map[batch_idx].float()  # [num_classes, 64, 64]

            loss_heat.append(
                self.local_focal(heat_pred_single.sigmoid(), gt_heat_map_single)
            )

            if gt_inst_mask[batch_idx].shape[0] > 0:
                gt_inst_mask_single = gt_inst_mask[batch_idx].float()
                gt_is_center_single = gt_is_center[batch_idx]
                kernel_pred_single = (
                    kernel_pred_single.permute(1, 2, 0)
                    .contiguous()
                    .view(
                        -1,
                        self.heads.ins_out_channels
                        * self.heads.kernel_size
                        * self.heads.kernel_size,  # [4096, 256]
                    )
                )
                kernel_pred_single = torch.cat(
                    [kernel_pred_single[gt_is_center_single]], 0
                ).view(
                    -1,
                    self.heads.ins_out_channels,
                    self.heads.kernel_size,
                    self.heads.kernel_size,  # [num_instance, 256, 1, 1]
                )
                ins_pred = F.conv2d(
                    feature_pred_single.unsqueeze(0), kernel_pred_single, stride=1
                ).view(-1, h, w)
                loss_inst.append(self.ins_loss(ins_pred, gt_inst_mask_single))

            else:
                continue

        losses = dict()
        if len(loss_inst) > 0:
            losses["loss_instance"] = torch.stack(loss_inst).mean()
        losses["loss_heatmap"] = torch.stack(loss_heat).mean()
        return losses

    def local_focal(self, pred, gt):
        """
        focal loss copied from CenterNet, modified version focal loss
        change log: numeric stable version implementation
        """
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # print(num_pos)

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def find_local_peak(self, heat, kernel=3):
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=(kernel) // 2
        )
        keep = (hmax == heat).float()
        return heat * keep
