import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
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

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger


@BACKBONES.register_module()
class RSSamVisionEncoder(BaseModule):
    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        peft_config=None,
        init_cfg=None,
    ):
        BaseModule.__init__(self)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = SamVisionEncoder(sam_config)
        # load checkpoint
        if init_cfg is not None:
            from mmcv.runner.checkpoint import load_checkpoint

            load_checkpoint(
                vision_encoder,
                init_cfg.get("checkpoint"),
                map_location="cpu",
                revise_keys=[(r"^module\.", ""), (r"^vision_encoder\.", "")],
                logger=get_root_logger(),
            )

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            raise ValueError("not support!")
            # config.update(peft_config)
            # peft_config = get_peft_config(config)
            # self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            # if is_main_process():
            #     self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        print("the vision encoder has been initialized")

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)
