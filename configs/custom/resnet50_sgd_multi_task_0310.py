# dataset settings
dataset_type = 'Multi_Task_Dataset'
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
    interval=1000, 
    metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'],
    save_best='mAP',
)

# training settings
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# fp16 settinjgs
# fp16 = dict(loss_scale='dynamic')

# checkpoint saving
checkpoint_config = dict(interval=100)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet', 
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskClsHead',
        num_tasks=[5, 9],
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss', 
            loss_weight=1.0,
        ),
    )
)

# basic settings
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(
        custom_keys=dict({'.backbone.classifier': dict(lr_mult=10)})))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[20, 50], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=100)
