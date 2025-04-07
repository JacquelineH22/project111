# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import Sequential
import numpy as np
from mmseg.ops import resize
from ..builder import HEADS

from .decode_head import BaseDecodeHead


def prob_2_entropy(prob):
    """convert probabilistic prediction maps to weighted self-information maps"""
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return F.binary_cross_entropy_with_logits(y_pred, y_truth_tensor)


@HEADS.register_module()
class DomainDiscriminator(Sequential):
    """
    Domain discriminator model from
    `ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation (CVPR 2019) <https://arxiv.org/abs/1811.12833>`_

    Distinguish pixel-by-pixel whether the input predictions come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        num_classes (int): num of classes in the predictions
        ndf (int): dimension of the hidden features

    Shape:
        - Inputs: :math:`(minibatch, C, H, W)` where :math:`C` is the number of classes
        - Outputs: :math:`(minibatch, 1, H, W)`
    """

    def __init__(self, num_classes, ndf=64):
        super(DomainDiscriminator, self).__init__(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        )
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_train(self, logits, domain_label="source", with_entropy=True, weight=1):
        assert domain_label in ["source", "target"]
        if with_entropy:
            probability = F.softmax(logits, dim=1)
            entropy = prob_2_entropy(probability)
        else:
            entropy = logits
        domain_prediciton = self.forward(entropy)
        losses = dict()
        if domain_label == "source":
            loss = bce_loss(domain_prediciton, 1)

        else:
            loss = bce_loss(domain_prediciton, 0)
        losses["domain_loss"] = loss * weight
        return losses
