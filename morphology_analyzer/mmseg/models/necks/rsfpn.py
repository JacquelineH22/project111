import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.cnn import ConvModule, build_norm_layer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoder,
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamModel,
    SamVisionEncoderOutput,
)
from typing import List, T, Tuple, Optional, Dict, Union

from mmseg.ops import resize
from ..builder import NECKS, MODELS


@NECKS.register_module()
class RSFPN(BaseModule):
    def __init__(
        self,
        feature_aggregator=None,
        feature_spliter=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, "feature_aggregator"):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, "feature_spliter"):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x  # 5层特征图 1/4， 1/8， 1/16， 1/32， 1/64


@MODELS.register_module()
class RSFeatureAggregator(BaseModule):
    in_channels_dict = {
        "facebook/sam-vit-base": [768] * (12 + 1),
        "facebook/sam-vit-large": [1024] * (24 + 1),
        "facebook/sam-vit-huge": [1280] * (32 + 1),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=256,
        select_layers=range(1, 12, 2),
        init_cfg=None,
        norm_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        self.in_channels = self.in_channels_dict[in_channels]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    build_norm_layer(norm_cfg, hidden_channels)[1],
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    build_norm_layer(norm_cfg, hidden_channels)[1],
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    build_norm_layer(norm_cfg, hidden_channels)[1],
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, "b h w c -> b c h w") for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))  # 改变通道数，不改变大小

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x


@MODELS.register_module()
class RSSimpleFPN(BaseModule):
    def __init__(
        self,
        backbone_channel: int,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(
                self.backbone_channel // 2, self.backbone_channel // 4, 2, 2
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2)
        )
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))  # 4x upsample
        inputs.append(self.fpn2(input))  # 2x upsample
        inputs.append(self.fpn3(input))  # Identity()
        inputs.append(self.fpn4(input))  # 2x max pool

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)  # 调整通道数，都为256
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
