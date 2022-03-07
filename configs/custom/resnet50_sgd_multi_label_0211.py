# dataset settings
dataset_type = 'Multi_Label_Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
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
        data_prefix='data/debug_black_blue_yellow_bus_car_suv',
        ann_file='data/debug_black_blue_yellow_bus_car_suv/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/debug_black_blue_yellow_bus_car_suv',
        ann_file='data/debug_black_blue_yellow_bus_car_suv/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/debug_black_blue_yellow_bus_car_suv',
        ann_file='data/debug_black_blue_yellow_bus_car_suv/meta/val.txt',
        pipeline=test_pipeline))
        
# evaluation = dict(
#     interval=1, 
#     metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'],
#     save_best='mAP',
# )

# training settings
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

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
        type='MultiLabelLinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0,
            reduction='mean'
        ),
    )
)

# basic settings
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(
        custom_keys=dict({'.backbone.classifier': dict(lr_mult=10)})))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=20, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=3)

