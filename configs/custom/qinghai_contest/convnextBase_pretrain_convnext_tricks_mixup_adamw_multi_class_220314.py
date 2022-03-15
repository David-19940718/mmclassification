_base_ = [
    '../../_base_/datasets/pipelines/color_rotate_aug.py'
]
# ---------- Model Setting ---------- #
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
            dict(
                type='Pretrained',
                checkpoint='weights/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth',
                prefix='backbone',
            )
        ]
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True),
        topk=(1,)
    ),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=1., num_classes=4, prob=1.)
    )
)


# ---------- Log Setting ---------- #
log_config = dict(
    interval=15,
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
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

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
work_dir = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/qinghai_contest/convnextBase_pretrain_convnext_tricks_mixup_adamw_multi_class_220314'
