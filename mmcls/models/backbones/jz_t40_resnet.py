import torch.nn as nn
from ..builder import BACKBONES
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops


'''Recommonded settings
IS_QUANTIZE = 1
BITW = 4
if BITW==8:
    BITA = 8
    WEIGHT_FACTOR = 3.0
    CLIP_MAX_VALUE = 6.0
elif BITW==4:
    BITA = 4
    WEIGHT_FACTOR = 3.0
    CLIP_MAX_VALUE = 4.0
elif BITW==2:
    BITA = 2

    
    WEIGHT_FACTOR = 2.0
    CLIP_MAX_VALUE = 2.0
else:
    BITA = 32
    WEIGHT_FACTOR = 3.
    CLIP_MAX_VALUE = 6.
TARGET_DEVICE = "T40"
'''

__all__ = [
    'ResNet', 
]

def T40Preprocess(target_device):
    return ops.Preprocess(target_device=target_device)


def T40Conv2D(
        in_channels, out_channels, 
        kernel_size=None, stride=1, pad=0, 
        groups=1, dilation=1, bias=False, bn=True, act=False, 
        first_layer=False, last_layer=False,
        is_quantize=False, bita=32, bitw=32, 
        clip_max_value=6.0, weight_factor=3.0, target_device="T40",
    ):
    assert(groups==1)
    assert(dilation==1)
    act_fn = nn.ReLU(inplace=True) if not is_quantize and act else None
    if isinstance(kernel_size, tuple):
        h = int(kernel_size[0])
        w = int(kernel_size[1])
    else:
        h = w = int(kernel_size)

    return ops.Conv2D(in_channels,
                      out_channels,
                      kernel_h = h,
                      kernel_w = w,
                      stride = stride,
                      activation_fn = act_fn,
                      enable_batch_norm = bn,
                      enable_bias = bias,
                      quantize = is_quantize,
                      first_layer = first_layer,
                      last_layer = last_layer,
                      padding = pad,
                      weight_bitwidth = bitw,
                      input_bitwidth = bita,
                      output_bitwidth = bita,
                      clip_max_value = clip_max_value,
                      weight_factor = weight_factor,
                      target_device = target_device)


def T40Add(
    channels,
    is_quantize=False,
    bita=32,
    target_device="T40",
):
    return ops.Shortcut(
        channels,
        quantize=is_quantize,
        input_bitwidth=bita,
        output_bitwidth=bita,
        target_device=target_device,
    )


def T40Maxpool(
        kernel_size, stride=2, padding=0, 
        target_device="T40",
    ):
    return ops.Maxpool2D(
        kernel_h=kernel_size, kernel_w=kernel_size, 
        stride=stride, padding=padding, 
        target_device=target_device,
    )


def conv3x3(
    in_planes, out_planes,
    stride=1, groups=1, dilation=1,
    is_quantize=False, bita=32, bitw=32, 
    clip_max_value=6.0, weight_factor=3.0, target_device="T40",
):
    return T40Conv2D(
        in_planes, out_planes, 
        kernel_size=3, stride=stride, pad=dilation, 
        groups=groups, dilation=dilation, 
        bias=False, bn=True, act=True,
        is_quantize=is_quantize, bita=bita, bitw=bitw, 
        clip_max_value=clip_max_value, 
        weight_factor=weight_factor, 
        target_device=target_device,
    )


def conv1x1(
    in_planes, out_planes, stride=1,
    is_quantize=False, bita=32, bitw=32, 
    clip_max_value=6.0, weight_factor=3.0, target_device="T40",
):
    return T40Conv2D(
        in_planes, out_planes, 
        kernel_size=1, stride=stride,
        bias=False, bn=True, act=True,
        is_quantize=is_quantize, bita=bita, bitw=bitw, 
        clip_max_value=clip_max_value, 
        weight_factor=weight_factor, 
        target_device=target_device,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 is_quantize=False, bita=32, bitw=32,
                 clip_max_value=6.0, weight_factor=3.0, target_device="T40",
                ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            inplanes, planes, stride,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor, 
            target_device=target_device,
        )
        self.conv2 = conv3x3(
            planes, planes,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor, 
            target_device=target_device,       
        )
        self.downsample = downsample
        self.stride = stride
        self.add = T40Add(
            planes,
            is_quantize=is_quantize,
            bita=bita,
            target_device=target_device,
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add([out, identity])

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 is_quantize=False, bita=32, bitw=32,
                 clip_max_value=6.0, weight_factor=3.0, target_device="T40",
                ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            inplanes, width,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor, 
            target_device=target_device,
        )
        self.conv2 = conv3x3(
            width, width, 
            stride=stride, 
            groups=groups, 
            dilation=dilation,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor, 
            target_device=target_device,  
        )
        self.conv3 = conv1x1(
            width, planes * self.expansion,
            is_quantize=is_quantize, 
            bita=bita, 
            bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor, 
            target_device=target_device, 
        )
        self.downsample = downsample
        self.stride = stride
        self.add = T40Add(
            planes * self.expansion,
            is_quantize=is_quantize,
            bita=bita,
            target_device=target_device,
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add([out, identity])

        return out


@BACKBONES.register_module()
class T40ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, is_quantize=False, bita=32, bitw=32,
                 clip_max_value=6.0, weight_factor=3.0, target_device="T40",
                ):
        super(T40ResNet, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        block, layers = self.arch_settings[depth]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.preprocess = T40Preprocess(target_device=target_device)

        self.conv1 = T40Conv2D(
            in_channels=3, out_channels=self.inplanes,
            kernel_size=3, stride=1, pad=1, 
            bias=False, bn=True, act=True,
            first_layer=True, last_layer=False,
            is_quantize=is_quantize, bita=bita, bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )
        self.maxpool = T40Maxpool(3, 2, 1, target_device=target_device)
        self.layer1 = self._make_layer(
            block, 64, layers[0],
            is_quantize=is_quantize, bita=bita, bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=1,
            dilate=replace_stride_with_dilation[0],
            is_quantize=is_quantize, bita=bita, bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1],
            is_quantize=is_quantize, bita=bita, bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2],
            is_quantize=is_quantize, bita=bita, bitw=bitw, 
            clip_max_value=clip_max_value, 
            weight_factor=weight_factor,
            target_device=target_device,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self, block, planes, blocks, stride=1, dilate=False,
            is_quantize=False, bita=32, bitw=32,
            clip_max_value=6.0, weight_factor=3.0, target_device="T40",
        ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion, stride,
                    is_quantize=is_quantize, bita=bita, bitw=bitw, 
                    clip_max_value=clip_max_value, 
                    weight_factor=weight_factor,
                    target_device=target_device,
                ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            is_quantize=is_quantize, bita=bita, bitw=bitw, 
                            clip_max_value=clip_max_value, 
                            weight_factor=weight_factor,
                            target_device=target_device,
                            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                is_quantize=is_quantize, bita=bita, bitw=bitw, 
                                clip_max_value=clip_max_value, 
                                weight_factor=weight_factor,
                                target_device=target_device,
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # input [224] -> ResNet18
        # torch.Size([16, 64, 112, 112])
        # torch.Size([16, 128, 112, 112])
        # torch.Size([16, 256, 56, 56])
        # torch.Size([16, 512, 28, 28])

        return x
