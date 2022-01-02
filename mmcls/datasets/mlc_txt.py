# Copyright (c) Jack.Wang All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class MLC(MultiLabelDataset):
    """Custom Multi-Label Classification Dataset."""

    CLASSES = ('cat', 'dog')

    def __init__(self, **kwargs):
        super(MLC, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.
        Annotations example -> 000001.txt:
        |-- cat 1
        |-- cat 1
        |-- dog 1
        |-- dog -1
        |-- ...
        Note 1 stands positive case while -1 stands difficult case follow by VOC.
        Returns:
            list[dict]: Annotation info from *.txt file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(self.ann_file)
        for img_id in img_ids:
            filename = f'images/{img_id}.jpg'
            txt_path = osp.join(self.data_prefix, 'labels',
                                f'{img_id}.xml')
            labels = []
            labels_difficult = []
            file = open(txt_path, 'r')
            for line in file.readlines():
                info = line.strip('\n')
                label_name, flag = filter(None, info.split(' '))
                if label_name not in self.CLASSES:
                    continue
                label = self.class_to_idx[label_name]
                difficult = True if int(flag) == -1 else False
                if difficult:
                    labels_difficult.append(label)
                else:
                    labels.append(label)
            file.close()
            gt_label = np.zeros(len(self.CLASSES))
            # The order cannot be swapped for the case where multiple objects
            # of the same kind exist and some are difficult.
            gt_label[labels_difficult] = -1
            gt_label[labels] = 1

            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)

        return data_infos
