# dataset settings
dataset_type = 'VOC'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, 
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

# checkpoint saving
checkpoint_config = dict(interval=5,)

# junzhen settings
is_quantize = 0
bitw = 32
if bitw==8:
    bita = 8
    weight_factor = 3.0
    clip_max_value = 6.0
elif bitw==4:
    bita = 4
    weight_factor = 3.0
    clip_max_value = 4.0
elif bitw==2:
    bita = 2
    weight_factor = 2.0
    clip_max_value = 2.0
else:
    bita = 32
    weight_factor = 3.
    clip_max_value = 6.
target_device = "T40"


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=512,
        loss=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0,
            reduction='mean'
        ),
    ))


# ---------- Schedules Setting ---------- #
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[90, 180, 270])
runner = dict(type='EpochBasedRunner', max_epochs=300)