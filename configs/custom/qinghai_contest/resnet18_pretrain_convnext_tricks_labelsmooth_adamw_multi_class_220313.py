_base_ = [
    '../../_base_/datasets/pipelines/color_rotate_aug.py'
]
# ---------- Model Setting ---------- #
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=2,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18',
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=512,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1,),
    )
)

# ---------- Log Setting ---------- #
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# ---------- Dataset Setting ---------- #
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 10,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_type = 'QingHai_Vehicle_Type'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/train',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/val',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/val',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContest/meta/val.txt',
        pipeline=test_pipeline))

# ---------- Training Setting ---------- #
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 32 * 2 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)

evaluation = dict(
    interval=1, 
    metric=['accuracy', 'recall', 'precision'],
    save_best='accuracy_top-1',
    metric_options={'topk': (1,)}
)
checkpoint_config = dict(interval=100)
runner = dict(type='EpochBasedRunner', max_epochs=300)
work_dir = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/qinghai_contest/resnet18_pretrain_convnext_tricks_labelsmooth_adamw_multi_class_220313'
