# Copyright (c) Jakc.Wang All rights reserved.
import mmcv
import numpy as np
import os.path as osp

from .builder import DATASETS
from .multi_task import MultiTaskDataset


@DATASETS.register_module()
class Multi_Task_Dataset(MultiTaskDataset):
    ATTRIBUTES = ['types', 'colors']
    CLASSES = [
        ['bus', 'car', 'suv', 'truck', 'van'],
        ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
    ]

    def __init__(self, **kwargs):
        super(Multi_Task_Dataset, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.
        Returns:
            list[dict]: Annotation info from json file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(self.ann_file)
        index2filename = self.get_index_to_filename()

        for img_id in img_ids:
            filename = f'images/{index2filename[img_id]}'
            jsonname = f'labels/{index2filename[img_id]}.json'
            json_path = osp.join(self.data_prefix, jsonname)
            json_info = self.load_json(json_path)
            labels = [
                json_info['step_1']['result'][0]['result'][name] for name in self.ATTRIBUTES
            ]
            indexs = [self.CLASSES[i].index(label) for i, label in enumerate(labels)]
            gt_label = [np.array(idx, dtype=np.int64) for idx in indexs]

            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label,
            )
            data_infos.append(info)
            '''
            For example:
                CLASSES = [['black', 'blue', 'yellow'], ['bus', 'car', 'suv']]
                labels = ['yellow', 'car']
                indexs = [2, 1]
                info = {
                    'img_prefix': 'data/0211_bit_multi_label_dataset',
                    'img_info': {'filename': 'images/000221487.jpg'},
                    'gt_label': [array([2]), array([1])], dtype=int64)
                }
            '''

        return data_infos
    
    
    def get_index_to_filename(self):
        import os
        index2filename = dict()
        image_path = osp.join(self.data_prefix, 'images')
        for filename in os.listdir(image_path):
            index2filename[os.path.splitext(filename)[0]] = filename
        return index2filename

    @staticmethod
    def load_json(p):
        import json
        """load *.json file.
        """
        with open(p, 'r') as f:
            info = f.read()
        return json.loads(info)

