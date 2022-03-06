import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class HUST_Vehicle_Color_Dataset(BaseDataset):
    CLASSES = ["yellow", "gray", "cyan", "white", "red", "black", "blue", "green"]
    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                print(np.array(gt_label, dtype=np.int64))
                exit(0)
                data_infos.append(info)
            return data_infos
