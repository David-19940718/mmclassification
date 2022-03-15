# Copyright (c) OpenMMLab. All rights reserved.
from ast import Gt
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import binary_cross_entropy


@LOSSES.register_module()
class WSLD(nn.Module):
    """PyTorch version of `Rethinking Soft Labels for Knowledge
    Distillation: A Bias-Variance Tradeoff Perspective
    <https://arxiv.org/abs/2102.00650>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        num_classes (int): Defaults to 1000.
    """

    def __init__(self, tau=1.0, loss_weight=1.0, num_classes=1000, task='multiclass'):
        super(WSLD, self).__init__()

        self.tau = tau
        self.task = task
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.norm = nn.Softmax(dim=1).cuda() if task=='multiclass' else nn.Sigmoid().cuda()
        self.lognorm = nn.LogSoftmax(dim=1).cuda() if task=='multiclass' else nn.LogSigmoid().cuda()
        self.hard_loss = nn.CrossEntropyLoss().cuda() if task=='multiclass' else binary_cross_entropy
        
    def forward(self, student, teacher):

        gt_labels = self.current_data['gt_label']

        # print(student)  # trorch.tensor.cuda with grad -> [num_samples, num_classes]
        # print(teacher)  # trorch.tensor.cuda without grad -> [num_samples, num_classes]
        # # tensor([3, 6, 0, 1, 6, 5, 6, 5, 7, 6, 0, 5, 5, 5, 6, 6], device='cuda:0')
        # print(gt_labels)
        
        student_logits = student / self.tau
        teacher_logits = teacher / self.tau

        teacher_probs = self.norm(teacher_logits)
        # tensor([[ 9.7203e-04,  9.4929e-03, -2.9162e-02, -4.5948e-03, -1.7229e-02,
                #-1.2977e-02,  1.5657e-02,  1.4110e-02],...]

        ce_loss = -torch.sum(
            teacher_probs * self.lognorm(student_logits), 1, keepdim=True)
        # tensor([[2.0800], ...*14, [2.0805]], device='cuda:0', grad_fn=<NegBackward>)
        # print(ce_loss)

        #  detach()用于切断反向传播
        student_detach = student.detach()
        teacher_detach = teacher.detach()
        log_norm_s = self.lognorm(student_detach)
        log_norm_t = self.lognorm(teacher_detach)
        
        # print(gt_labels)
        # tensor([6, 5, 5, 0, 0, 0, 5, 6, 0, 5, 4, 5, 3, 4, 3, 5], device='cuda:0')
        if self.task == 'multiclass':
            one_hot_labels = F.one_hot(
                gt_labels, num_classes=self.num_classes).float()
        elif self.task == 'multilabel':
            one_hot_labels = gt_labels.type_as(student)
        else:
            raise ValueError("Invalid value.")
        '''
        tensor([[0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.]], device='cuda:0')
        '''
        # 分别计算student和teacher相对真值的loss
        ce_loss_s = -torch.sum(one_hot_labels * log_norm_s, 1, keepdim=True)
        ce_loss_t = -torch.sum(one_hot_labels * log_norm_t, 1, keepdim=True)

        focal_weight = ce_loss_s / (ce_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(-focal_weight)
        # print(focal_weight) # tensor([[0.6288],...,[0.6322]], device='cuda:0')
        ce_loss = focal_weight * ce_loss
        # print(ce_loss) # tensor([[1.3079],...,[1.3148]], device='cuda:0', grad_fn=<MulBackward0>)
        loss = (self.tau**2) * torch.mean(ce_loss)
        # print(loss)  # tensor(5.2458, device='cuda:0', grad_fn=<MulBackward0>)

        if self.task == 'multiclass':
            hard_loss = self.hard_loss(student, gt_labels)
        elif self.task == 'multilabel':
            gt_labels = gt_labels.type_as(student)
            hard_loss = self.hard_loss(student, gt_labels)
        # print(hard_loss)  # tensor(0.6876, device='cuda:0', grad_fn=<MeanBackward0>)

        # print(self.loss_weight)  # 2.5

        loss = self.loss_weight * loss + hard_loss
        # print(loss)  # tensor(13.1144, device='cuda:0', grad_fn=<MulBackward0>)

        return loss
