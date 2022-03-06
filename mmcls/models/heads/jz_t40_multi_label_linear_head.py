import torch
import torch.nn as nn

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops


def T40Flatten(shape_list, target_device="T40"):
    return ops.Flatten(shape_list, target_device=target_device)


def T40FullConnected(
        in_channels, out_channels, 
        bias=True, bn=False, act=False, 
        last_layer=False,
        is_quantize=False, bita=32, bitw=32,
        clip_max_value=6.0, 
        weight_factor=3.0,
        target_device="T40"
    ):
    act_fn = nn.ReLU(inplace=True) if not is_quantize and act else None
    if(last_layer):
        assert(act_fn == None)
    return ops.FullConnected(in_channels,
                             out_channels,
                             activation_fn=act_fn,
                             enable_batch_norm=bn,
                             enable_bias=bias,
                             quantize = is_quantize,
                             last_layer=last_layer,
                             weight_bitwidth=bitw,
                             input_bitwidth=bita,
                             output_bitwidth=32 if last_layer else bita,
                             clip_max_value=clip_max_value,
                             weight_factor=weight_factor,
                             target_device=target_device,
                            )


@HEADS.register_module()
class T40MultiLabelLinearClsHead(MultiLabelClsHead):
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
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 is_quantize=False, bita=32, bitw=32,
                 clip_max_value=6.0, 
                 weight_factor=3.0,
                 target_device="T40",
                ):
        super(T40MultiLabelLinearClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.flatten = T40Flatten([-1, 512], target_device)
        self.fc = T40FullConnected(
            in_channels, 
            num_classes,
            bias=True, 
            bn=False, 
            act=False, 
            last_layer=True,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw,
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )

    def forward_train(self, x, gt_label, **kwargs):
        
        gt_label = gt_label.type_as(x[0])
        # output.shape = [batch_size, num_classes]
        cls_score = self.fc(self.flatten(x))
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        cls_score = self.fc(self.flatten(x))

        if sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
