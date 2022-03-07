# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 num_tasks,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiTaskClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        assert isinstance(num_tasks, list), \
            f"type(num_classes) must be list, but got {type(num_tasks)}."
        for i, num in enumerate(num_tasks):
            if num <= 0:
                raise ValueError(
                    f'num_tasks[{i}]={num} must be a positive integer')

        self.in_channels = in_channels
        self.num_tasks = num_tasks

        self.fc_layers = nn.ModuleList()
        for n in num_tasks:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(self.in_channels, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, n)
                )
            )

        self.test = nn.Linear(self.in_channels, self.num_tasks[0])


    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        losses = None
        for n in range(len(self.num_tasks)):
            x = self.pre_logits(x)
            cls_score = self.fc_layers[n](x)
            gt_score = gt_label[n]
            loss = self.loss(cls_score, gt_score, **kwargs)
            if losses:
                losses = torch.stack([losses, loss['loss']])
            else:
                losses = loss['loss']
        losses = {'loss': torch.mean(losses)}
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
        pred = []
        for n in range(len(self.num_tasks)):
            x = self.pre_logits(x)
            cls_score = self.fc_layers[n](x)
            if softmax:
                pred.append(
                    (F.softmax(cls_score, dim=1) if cls_score is not None else None)
                )
            else:
                pred.append(cls_score)

        if post_process:
            return [self.post_process(p) for p in pred]
        else:
            return pred



