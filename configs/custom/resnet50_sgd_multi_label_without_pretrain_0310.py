_base_ = ['./resnet50_sgd_multi_label_0310.py']

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet', 
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
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