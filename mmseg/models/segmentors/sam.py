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


class SAM(SamModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, hf_pretrain_name, init_cfg=None):
        # build SAM
        config = SamConfig.from_pretrained(hf_pretrain_name)
        super().__init__(config)
        self.init_cfg = init_cfg
        self._is_init = False
        # if self.init_cfg is not None:
        #     load_checkpoint(
        #         self,
        #         self.init_cfg.get("checkpoint"),
        #         map_location="cpu",
        #         revise_keys=[(r"^module\.", ""), (r"^sam\.", "")],
        #     )
        #     self._is_init = True
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False 

    def post_init(self):
        pass

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
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cell_num=None,
        prompt_pred=None,
        **kwargs,
    ):
        """
        修改SamModel的mask_decoder的forward()，其他部分保持一致
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

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError(
                "Only one of pixel_values and image_embeddings can be provided."
            )

        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
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

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(
                input_points[:, :, :, 0], dtype=torch.int, device=input_points.device
            )

        if prompt_pred is None:
            # sparse_embeddings[num, 1, 2, 256], num=所有图片的mask总和,即cell_num之和
            # dense_embeddings [num, 256, 16, 16]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                input_masks=input_masks,
            )
        else:
            batch_size, point_batch_size = input_points.shape[:2]
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(
                batch_size,
                -1,
                self.prompt_encoder.image_embedding_size[0],
                self.prompt_encoder.image_embedding_size[1],
            )
            input_points = input_points.reshape(-1, 2).long()  # 256*256图片的坐标
            input_points = input_points // 4  # 转为64*64的坐标
            prompt_pred = prompt_pred.repeat_interleave(cell_num, dim=0)  # 64*64
            prompt_channels = prompt_pred.shape[1]
            sparse_embeddings = torch.zeros(
                (input_points.shape[0], prompt_channels),
                device=prompt_pred.device,
                dtype=torch.float32,
            )
            for i in range(input_points.shape[0]):
                x, y = input_points[i]
                sparse_embeddings[i] = prompt_pred[i, :, y, x]
            sparse_embeddings = sparse_embeddings.reshape(
                input_points.shape[0], 1, -1, 256
            )
            sparse_embeddings = (
                sparse_embeddings
                + self.prompt_encoder.point_embed[1].weight[None, None, :, :]
            )

        # modify mask decoder
        _, num_channels, height, width = image_embeddings.shape
        batch_size = sparse_embeddings.shape[0]    # 这是这个batch中所有图像的目标点数量之和
        point_batch_size = sparse_embeddings.shape[1]  # should be 1
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight],
            dim=0,
        )  # [5, 256]
        output_tokens = output_tokens.repeat(
            batch_size, point_batch_size, 1, 1
        )  # [num, 1, 5, 256]

        if sparse_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.mask_decoder.iou_token.weight.dtype)

        """
        上面可能不用改
        """
        # [num, 256, 16, 16]
        image_embeddings_re = image_embeddings.repeat_interleave(cell_num, dim=0)   #是为了让image_embedding和sparse_embedding能保持维度一样，做并行
        image_embeddings_re = image_embeddings_re + dense_embeddings
        # [num, 256, 16, 16]
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            cell_num, 0
        )

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings_re, attentions = self.mask_decoder.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings_re,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[
            :, :, 1 : (1 + self.mask_decoder.num_mask_tokens), :
        ]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings_re = image_embeddings_re.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.mask_decoder.upscale_conv1(image_embeddings_re)
        upscaled_embedding = self.mask_decoder.activation(
            self.mask_decoder.upscale_layer_norm(upscaled_embedding)
        )
        upscaled_embedding = self.mask_decoder.activation(
            self.mask_decoder.upscale_conv2(upscaled_embedding)
        )

        hyper_in_list = []
        for i in range(self.mask_decoder.num_mask_tokens):
            current_mlp = self.mask_decoder.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(
            batch_size, point_batch_size, num_channels, height * width
        )
        masks = (hyper_in @ upscaled_embedding).reshape(
            batch_size, point_batch_size, -1, height, width
        )

        # Generate mask quality predictions
        iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        low_res_masks, iou_predictions, mask_decoder_attentions = outputs

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=image_embeddings,  # 返回image_embeddings，用于交互
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )
