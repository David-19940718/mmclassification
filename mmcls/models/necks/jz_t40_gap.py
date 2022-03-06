import torch.nn as nn

from ..builder import NECKS
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops


def T40GAP(
    channels, 
    keepdim=True, 
    last_layer=False,
    is_quantize=False,
    bita=32,
    target_device="T40",
):
    return ops.AdaptiveAvgpool2D(channels,
                                 keepdim=keepdim,
                                 quantize=is_quantize,
                                 input_bitwidth=bita,
                                 output_bitwidth=32 if last_layer else bita,
                                 last_layer=last_layer,
                                 target_device=target_device,
                                )

@NECKS.register_module()
class T40GlobalAveragePooling(nn.Module):
    """T40 Global Average Pooling neck.
    """
    def __init__(
        self, 
        in_channels,
        is_quantize=False,
        bita=32,
        target_device="T40",
    ):
        super(T40GlobalAveragePooling, self).__init__()
        self.gap = T40GAP(
            in_channels,
            is_quantize=is_quantize,
            bita=bita,
            target_device=target_device,
        )

    def init_weights(self):
        pass

    def forward(self, inputs):
        outs = self.gap(inputs)
        return outs