# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class MultiTaskClsHead(ClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiTaskClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        assert isinstance(num_classes, list), \
            f"type(num_classes) must be list, but got {type(num_classes)}."
        for i, num in enumerate(num_classes):
            if num <= 0:
                raise ValueError(
                    f'num_classes[{i}]={num} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc_list = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.in_channels, 128), nn.ReLU(inplace=True), nn.Linear(128, n),]) for n in num_classes
        ])


    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        losses = []
        for n in range(len(self.num_classes)):
            x = self.pre_logits(x)
            gt_label = gt_label[n].type_as(x)
            cls_score = self.fc_list[n](x)
            losses.append(self.loss(cls_score, gt_label, **kwargs))
        losses = torch.mean(losses)
        return losses

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        print("sadasd")
        exit(0)
        
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred



