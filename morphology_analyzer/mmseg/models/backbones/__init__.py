# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones

from .mix_transformer import (
    MixVisionTransformer,
    mit_b0,
    mit_b1,
    mit_b2,
    mit_b3,
    mit_b4,
    mit_b5,
)
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .unet import UNet
from .samvisionencoder import SamVisionEncoder
from .swin import SwinTransformer
from .prompter import Prompter
from .prompter2 import Prompter2
from .prompter3 import Prompter3
from .prompter4 import Prompter4
from .vit import ViT
from .hrformer import HRFormer

__all__ = [
    "ResNet",
    "ResNetV1d",
    "ResNeXt",
    "ResNeSt",
    "MixVisionTransformer",
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
    "mit_b5",
    "UNet",
    "SamVisionEncoder",
    "SwinTransformer",
    "Prompter",
    "Prompter2",
    "Prompter3",
    "Prompter4",
    "ViT",
    "HRFormer",
]
