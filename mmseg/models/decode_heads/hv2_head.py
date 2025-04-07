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
class HV2Head(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(HV2Head, self).__init__(input_transform="multiple_select", **kwargs)

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

        self.criterion2 = nn.BCEWithLogitsLoss(reduction="none")

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

        x = self.fusion_conv(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)
        # x = F.tanh(x)
        return x

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        gt_hv_map,
        train_cfg,
        seg_weight=None,
    ):
        # bs ,3, h, w
        # 前两通道是HV，最后通道是seg
        seg_logit = self.forward(inputs)

        seg_logit = resize(
            input=seg_logit,
            size=gt_semantic_seg.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        hv_logit = seg_logit[:, :2].permute(0, 2, 3, 1).contiguous()
        mse_loss = self.mse_loss(gt_hv_map, hv_logit, seg_weight)
        focus = (gt_semantic_seg.squeeze(1) == 1).type(torch.int)
        msge_loss = self.msge_loss(gt_hv_map, hv_logit, focus)

        loss_seg = self.criterion2(seg_logit[:, 2:], gt_semantic_seg.float())
        loss_mask = gt_semantic_seg != 255
        loss_masked = loss_seg.where(loss_mask, torch.tensor(0.0).to(seg_logit.device))
        loss_seg = loss_masked.sum() / loss_mask.sum()

        losses = dict()
        losses["loss_hv_mse"] = mse_loss
        losses["loss_hv_msge"] = msge_loss
        losses["loss_seg"] = loss_seg

        return losses

    def mse_loss(self, true, pred, weight):
        """Calculate mean squared error loss.

        Args:
            true: ground truth of combined horizontal
                  and vertical maps
                  (bs, h, w, 2)
            pred: prediction of combined horizontal
                  and vertical maps
                  (bs, h, w, 2)
            weight: (bs, h, w)
        Returns:
            loss: mean squared error

        """
        if weight is not None:
            weight = weight.unsqueeze(-1)
            weight = weight.expand((2, pred.shape[1], pred.shape[2], 2))
        loss = pred - true
        loss = (loss * loss).mean()
        return loss

    ####
    def msge_loss(self, true, pred, focus):
        """Calculate the mean squared error of the gradients of
        horizontal and vertical map predictions. Assumes
        channel 0 is Vertical and channel 1 is Horizontal.

        Args:
            true:  ground truth of combined horizontal
                   and vertical maps
                   (bs, h, w, 2)
            pred:  prediction of combined horizontal
                   and vertical maps
                   (bs, h, w, 2)
            focus: area where to apply loss (we only calculate
                    the loss within the nuclei)
                    (bs, h, w)  0表示背景，1表示细胞核

        Returns:
            loss:  mean squared error of gradients

        """

        def get_sobel_kernel(size):
            """Get sobel kernel with a given size."""
            assert size % 2 == 1, "Must be odd, get size=%d" % size

            h_range = torch.arange(
                -size // 2 + 1,
                size // 2 + 1,
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
            v_range = torch.arange(
                -size // 2 + 1,
                size // 2 + 1,
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
            h, v = torch.meshgrid(h_range, v_range)
            kernel_h = h / (h * h + v * v + 1.0e-15)
            kernel_v = v / (h * h + v * v + 1.0e-15)
            return kernel_h, kernel_v

        ####
        def get_gradient_hv(hv):
            """For calculating gradient."""
            kernel_h, kernel_v = get_sobel_kernel(5)
            kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
            kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

            h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
            v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

            # can only apply in NCHW mode
            h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
            v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
            dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
            dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
            return dhv

        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1)
        true_grad = get_gradient_hv(true)
        pred_grad = get_gradient_hv(pred)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss
