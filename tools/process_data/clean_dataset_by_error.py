import os
import shutil
from tqdm import tqdm

from tools.custom_tools.utils import txt_to_list

def v1():
    """
    Remove dataset by error list
    """
    save_path = '/data/workspace_jack/vehicle_attribute_dataset/unconfirm'
    error_path = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/eval/convnext_xlarge_adamw_pretrain_multi_label_0306_train_val/error_list.txt'
    error_list = txt_to_list(error_path)
    for src in tqdm(error_list):
        filename = os.path.basename(src)
        dst = os.path.join(save_path, filename)
        shutil.move(src, dst)
        shutil.move(src+'.json', dst+'.json')


if __name__ == '__main__':
    v1()
