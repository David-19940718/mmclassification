# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import (DATASETS, PIPELINES, SAMPLERS, build_dataloader,
                      build_dataset, build_sampler)
from .cifar import CIFAR10, CIFAR100
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               RepeatDataset, KFoldDataset)
from .imagenet import ImageNet
from .imagenet21k import ImageNet21k
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .samplers import DistributedSampler, RepeatAugSampler
from .voc import VOC
from .hust_vehicle_color import HUST_Vehicle_Color_Dataset
from .bit_vehicle_type import BIT_Vehicle_Type_Dataset
from .bit_vehicle_color import BIT_Vehicle_Color_Dataset
from .bit_vehicle_color_type import BIT_Color_Type_Dataset
from .multi_label_dataset import Multi_Label_Dataset
from .multi_task_dataset import Multi_Task_Dataset
from .gender import Gender_Dataset
from .qinghai_vehicle_type import QingHai_Vehicle_Type

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'MultiLabelDataset', 'build_dataloader', 'build_dataset',
    'DistributedSampler', 'ConcatDataset', 'RepeatDataset', 'KFoldDataset',
    'ClassBalancedDataset', 'DATASETS', 'PIPELINES', 'ImageNet21k', 'SAMPLERS',
    'build_sampler', 'RepeatAugSampler', 
    'HUST_Vehicle_Color_Dataset', 'Gender_Dataset', 'QingHai_Vehicle_Type',
    'BIT_Vehicle_Type_Dataset', 'BIT_Vehicle_Color_Dataset', 'BIT_Color_Type_Dataset', 
    'Multi_Label_Dataset', 'Multi_Task_Dataset', 
]
