# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

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
            **kwargs
        )
    elif type == "conv":
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs["kernel_size"] // 2,
            **kwargs
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
class CellPoseHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(CellPoseHead, self).__init__(input_transform="multiple_select", **kwargs)

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
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

        self.criterion = nn.MSELoss(reduction="mean")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, inputs, return_last_feat=False):
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

        x = self.fusion_conv(torch.cat(list(_c.values()), dim=1))
        out = self.cls_seg(x)
        # x = F.tanh(x)
        if return_last_feat:
            return x, out
        return out

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        gt_vec,
        train_cfg,
        seg_weight=None,
        return_last_feat=False,
        return_logits=False,
        dummy_gt_semantic_seg=None,
    ):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            gt_cellpose_map (Tensor)
            train_cfg (dict): The training config.
            seg_weight (int)
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if return_last_feat:
            # bs ,3, h, w
            last_feat, seg_logit = self.forward(
                inputs, return_last_feat=return_last_feat
            )
            losses["last_feat"] = last_feat
        else:
            # bs ,3, h, w
            seg_logit = self.forward(inputs, return_last_feat=return_last_feat)

        seg_logit = resize(
            input=seg_logit,
            size=gt_semantic_seg.shape[2:]
            if gt_semantic_seg is not None
            else dummy_gt_semantic_seg.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if return_logits:
            losses["logits"] = seg_logit

        # target data没有gt
        if gt_semantic_seg is None:
            return losses
        # bs, h, w, 2
        vec = 5.0 * gt_vec
        # bs, 2, h, w
        vec = vec.permute(0, 3, 1, 2).contiguous()
        loss_vec = self.criterion(seg_logit[:, :2], vec) / 2
        loss_seg = self.criterion2(seg_logit[:, 2:], gt_semantic_seg.float())

        losses["loss_vec"] = loss_vec
        losses["loss_seg"] = loss_seg

        return losses
