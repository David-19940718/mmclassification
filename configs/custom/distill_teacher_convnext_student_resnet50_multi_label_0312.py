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
        pipeline=test_pipeline)
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

evaluation = dict(
    interval=1, 
    metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'],
    save_best='mAP',
)

# training settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
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
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet', 
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        # frozen_stages=2,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='torchvision://resnet50',
        # )
    ),
    neck=dict(type='GlobalAveragePooling'),
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

teacher = dict(
    type='mmcls.ImageClassifier',
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

# distill settings
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='WSLD',
                        name='loss_wsld',
                        tau=2,
                        loss_weight=2.5,
                        num_classes=14)
                ])
        ]),
)

find_unused_parameters = True

