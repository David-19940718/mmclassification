# ---------- Model Setting ---------- #
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        # frozen_stages=2,  # 冻结前两层参数
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50',
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),))

# ---------- Training Setting ---------- #
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[10, 20])
runner = dict(type='EpochBasedRunner', max_epochs=30)

# checkpoint saving
checkpoint_config = dict(interval=5,)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# ---------- Schedules Setting ---------- #
# optimizer
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


# ---------- Dataset Setting ---------- #
dataset_type = 'Gender_Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=(128, 64), padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=(256, 128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/train',
        ann_file='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/val',
        ann_file='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/val',
        ann_file='/data/workspace_robert/par_dataset/Market-1501-v15.09.15/v1/mm/gender/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, 
    metric=['accuracy', 'recall', 'precision'],
    save_best='accuracy_top-1',
    metric_options={'topk': (1,)}
)
