data_root = "/data1/hounaiqiao/SpatioCell/datasets/inst/cpm17/preprocessed/"
test_image_dir = "/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/cpm17/test/image/"
test_label_dir = "/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/cpm17/test/label/"
work_dir = "/data1/hounaiqiao/SpatioCell/work_dirs/inst/cpm17/train-prompt/"

num_classes = 2

hf_sam_pretrain_ckpt_path = "/data1/hounaiqiao/SpatioCell/work_dirs/inst/cpm17/train-sam/latest.pth"
hf_sam_pretrain_name = "checkpoints/sam-vit-huge"
mit_b3_pretrain_ckpt_path = "checkpoints/mit_b3.pth"

import cv2

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
norm_cfg = dict(type="BN", requires_grad=True)
find_unused_parameters = True
num_neg_prompt = 1

model = dict(
    type="PromptSam",
    num_neg_prompt=0,
    train_sam=False,
    train_prompt=True,
    prompter=dict(
        type="Prompter2",
        with_mse=False,
        alpha=2,
        beta=4,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=3.0),
        ],
        backbone=dict(
            type="mit_b3",
            style="pytorch",
            pretrained=mit_b3_pretrain_ckpt_path,
        ),
        num_classes=num_classes,
        in_index=[0, 1, 2, 3],
        in_channels=[64, 128, 320, 512],
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type="mlp", act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type="mlp", act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type="aspp",
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type="ReLU"),
                norm_cfg=norm_cfg,
            ),
        ),
        pointnu_head_cfg=dict(
            seg_feat_channels=256,
            ins_out_channels=256,
            stacked_convs=7,
            kernel_size=1,
            norm_cfg=norm_cfg,
        ),
    ),
    sam_model=dict(
        type="SAM",
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(checkpoint=hf_sam_pretrain_ckpt_path, type="Pretrained"),
    ),
    test_cfg=dict(
        mode="slide",
        crop_size=(256, 256),
        stride=(164, 164),
        score_thr=0.1,
        nms_pre=300,
        max_per_img=150,  # maximum objects per image
        update_thr=0.3,
        iou_thr=0.7,
    ),
)

crop_size = (256, 256)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotationsNpy"),
    dict(
        type="Albu",
        transforms=[
            dict(
                type="Affine",
                scale=(0.75, 1.5),
                translate_percent=0.1,
                rotate=(-180, 180),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                cval=0,
                cval_mask=0,
                p=0.8,
            ),
        ],
        keymap=dict(
            img="image",
            gt_instance="mask",
        ),
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    dict(type="RandomFlip", direction="vertical", prob=0.5),
    dict(
        type="Albu",
        transforms=[
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2,
            ),
            dict(
                type="HueSaturationValue",
                p=0.2,
            ),
            dict(
                type="OneOf",
                transforms=[
                    dict(type="Blur", blur_limit=3, p=1.0),
                    dict(type="MedianBlur", blur_limit=3, p=1.0),
                    dict(type="GaussNoise", p=1.0),
                ],
                p=0.2,
            ),
        ],
    ),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(
        type="GetHeatMap",
        size=crop_size,
        num_classes=2,
        grid_size=crop_size[0] // 4,
        num_neg_prompt=num_neg_prompt,
        min_area=0,
        fix_mirror=True,
    ),
    dict(type="GetSemanticSeg"),
    dict(type="Normalize99"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_semantic_seg",
            "gt_heat_map",
            "gt_inst_mask",
            "gt_is_center",
            "center_xy",
            "gt_sam_inst_masks",
            "gt_sam_prompt_points",
            "gt_sam_prompt_labels",
            "cell_num",
        ],
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ),
    ),
]

valid_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotationsNpy"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=None,
        img_ratios=[
            1.0,
        ],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize99"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=[
                    "img",
                ],
                meta_keys=(
                    "filename",
                    "ori_filename",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "img_norm_cfg",
                ),
            ),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=None,
        img_ratios=[
            1.0,
        ],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize99"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=[
                    "img",
                ],
                meta_keys=(
                    "filename",
                    "ori_filename",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "img_norm_cfg",
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type="NucleiPointnuDataset",
        data_root=data_root,
        img_dir="images/train",
        ann_dir="labels/train",
        pipeline=train_pipeline,
        img_suffix=".png",
        seg_map_suffix=".npy",
    ),
    val=dict(
        type="NucleiPointnuDataset",
        data_root=data_root,
        img_dir="images/valid",
        ann_dir="labels/valid",
        pipeline=valid_pipeline,
        img_suffix=".png",
        seg_map_suffix=".npy",
    ),
    test=dict(
        type="NucleiPointnuDataset",
        data_root= "",
        img_dir=test_image_dir,
        ann_dir=test_label_dir,
        pipeline=test_pipeline,
        img_suffix=".png",
        seg_map_suffix=".mat",
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
seed = 64

runner = dict(type="EpochBasedRunner", max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1, save_optimizer=False)
evaluation = dict(interval=4, metric=["mIoU", "mDice"], by_epoch=True)
