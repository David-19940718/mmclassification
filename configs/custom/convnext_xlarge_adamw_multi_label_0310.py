# dataset settings

dataset_type = 'Multi_Label_Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/0310_train_standford_compcars_bus_test_benchmarkv1',
        ann_file='data/0310_train_standford_compcars_bus_test_benchmarkv1/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/0310_train_standford_compcars_bus_test_benchmarkv1',
        ann_file='data/0310_train_standford_compcars_bus_test_benchmarkv1/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/0310_train_standford_compcars_bus_test_benchmarkv1',
        ann_file='data/0310_train_standford_compcars_bus_test_benchmarkv1/meta/test.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=50, 
    metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'],
    save_best='mAP',
)

# training settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/convnext_xlarge_adamw_pretrain_multi_label_0306'
load_from = None
resume_from = None
workflow = [('train', 1)]

# checkpoint saving
checkpoint_config = dict(interval=10,)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='xlarge',
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
                checkpoint='weights/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth',
                prefix='backbone',
            )
        ]),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=14,
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0,
            reduction='mean'
        ),
    )
)

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
    lr=5e-4 * 16 * 2 / 512,
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

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
runner = dict(type='EpochBasedRunner', max_epochs=100)
