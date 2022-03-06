# Copyright (c) Jakc.Wang All rights reserved.
import mmcv
import numpy as np
import os.path as osp

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class Multi_Label_Dataset(MultiLabelDataset):
    CLASSES = (
        'bus',
        'car',
        'suv',
        'truck',
        'van',
        'black',
        'blue',
        'coffee',
        'gray',
        'green',
        'orange',
        'red',
        'white',
        'yellow',
    )

    def __init__(self, **kwargs):
        super(Multi_Label_Dataset, self).__init__(**kwargs)

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
                label_name for label_name in json_info['step_1']['result'][0]['result'].values()
            ]
            gt_label = np.zeros(len(self.CLASSES))
            indexs = [self.class_to_idx[label] for label in labels]
            gt_label[indexs] = 1
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)
            '''
            For example:
                {'car': 0, 'suv': 1, 'van': 2, 'black': 3, 'blue': 4, 'coffee': 5, 'gray': 6, 'green': 7, 'orange': 8, 'red': 9, 'white': 10, 'yellow': 11}
                labels = ['gray', 'van']
                indexs = [6, 2]
                info = {
                    'img_prefix': 'data/0211_bit_multi_label_dataset',
                    'img_info': {'filename': 'images/000221487.jpg'},
                    'gt_label': array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int8)
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
