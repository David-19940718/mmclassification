# ---------- Model Setting ---------- #
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
        # topk=(1,),
    )


# ---------- Training Setting ---------- #
# checkpoint saving
checkpoint_config = dict(interval=5, CLASSES=["SUV", "Van", "Car"])
# yapf:disable
log_config = dict(
    interval=100,
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


# ---------- Schedules Setting ---------- #
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[60, 75, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)


# ---------- Dataset Setting ---------- #
dataset_type = 'BIT_Vehicle_Type_Dataset'
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
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/train',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/val',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/val',
        ann_file='/home/jack/Projects/openmmlab/mmclassification/data/0208_vehicle_attr_dataset/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, 
    metric=['accuracy', 'recall', 'precision'],
    save_best='accuracy_top-1',
    metric_options={'topk': (1,)}
)
