from ..builder import SEGMENTORS
from .encoder_decoder_cellpose import EncoderDecoderCellPose


@SEGMENTORS.register_module()
class EncoderDecoderHV2(EncoderDecoderCellPose):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(EncoderDecoderHV2, self).__init__(
            backbone,
            decode_head,
            neck,
            auxiliary_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
        )

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg=None,
        seg_weight=None,
        gt_hv_map=None,
        return_feat=False,
        mode="dec",
        **kwargs
    ):
        assert mode in ["all", "aux", "dec"]
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses["features"] = x
        if mode == "all" or mode == "dec":
            loss_decode = self._decode_head_forward_train(
                x,
                img_metas,
                gt_semantic_seg,
                gt_hv_map,
                seg_weight,
            )
            losses.update(loss_decode)

        if self.with_auxiliary_head and (mode == "all" or mode == "aux"):
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight, **kwargs
            )
            losses.update(loss_aux)

        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ["slide", "whole", "slide2"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            seg_logit = self.slide_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == "whole":
            seg_logit = self.whole_inference(img, img_meta, rescale)
        else:
            seg_logit = self.slide_inference2(img, img_meta, rescale)

        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                seg_logit = seg_logit.flip(dims=(3,))  # 第0通道是水平距离
                seg_logit[:, 0, ...] = -seg_logit[:, 0, ...]
            elif flip_direction == "vertical":  # 第1通道是垂直距离
                seg_logit = seg_logit.flip(dims=(2,))
                seg_logit[:, 1, ...] = -seg_logit[:, 1, ...]
        return seg_logit
