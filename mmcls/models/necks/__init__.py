# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .jz_t40_gap import T40GlobalAveragePooling

__all__ = ['GlobalAveragePooling', 'T40GlobalAveragePooling']
