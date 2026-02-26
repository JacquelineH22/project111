from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoder,
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamModel,
    SamVisionEncoderOutput,
    SamPreTrainedModel,
    SamImageSegmentationOutput,
)
from mmcv.runner import load_checkpoint, BaseModule

from mmseg.utils import get_root_logger


class SAM_Semantic(SamModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, hf_pretrain_name, num_classes, init_cfg=None):
        # 初始化 SAM 模型
        config = SamConfig.from_pretrained(hf_pretrain_name)
        super().__init__(config)
        self.num_classes = num_classes  # 多分类任务的类别数量
        self.init_cfg = init_cfg
        self._is_init = False

        # 冻结 prompt_encoder 权重
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False 

        # 修改最后的分类头，替换 mask decoder 的输出为多分类的类别预测
        self.semantic_head = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def init_weights(self) -> None:
        if not self._is_init and self.init_cfg is not None:
            logger = get_root_logger()
            load_checkpoint(
                self,
                self.init_cfg.get("checkpoint"),
                map_location="cpu",
                logger=logger,
                revise_keys=[(r"^module\.", ""), (r"^sam\.", "")],
            )
            self._is_init = True

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        将 SAM 的 forward() 方法修改为语义分割，输出每个像素的类别。
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None and image_embeddings is None:
            raise ValueError(
                "Either pixel_values or image_embeddings must be provided."
            )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        batch_size = (
            pixel_values.shape[0]
            if pixel_values is not None
            else image_embeddings.shape[0]
        )
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        vision_attentions = None
        vision_hidden_states = None

        # 如果提供了 pixel_values，通过 vision_encoder 生成图像特征
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        # 根据 mask decoder 的修改，直接生成多分类的类别概率图
        image_embeddings = image_embeddings + image_positional_embeddings

        # 使用修改后的语义分割头生成分类预测
        semantic_predictions = self.semantic_head(image_embeddings)

        if not return_dict:
            return semantic_predictions

        return SamImageSegmentationOutput(
            pred_masks=semantic_predictions,  # 在语义分割中，每个像素的分类
            vision_hidden_states=image_embeddings,
            vision_attentions=vision_attentions,
        )
