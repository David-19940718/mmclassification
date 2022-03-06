import os
import json
import shutil
import random

from tqdm import tqdm
from loguru import logger

from tools.custom_tools.utils import load_json, save_json, mkdir, list_to_txt


"""
根据generate_stat_info生成多标签目录格式
|-- demo
    |-- images
    |-- labels [遵循LabelBee格式]
    |-- meta
        |-- train.txt
            |-- 000000
            |-- 000001
            ...
        |-- val.txt
            |-- 000010
            |-- 000011
            ...
"""


def split_train_val_list(total_list, split_ratio=0.8):
    random.shuffle(total_list)
    img_len = int(split_ratio * len(total_list))
    train_list = total_list[:img_len]
    val_list = total_list[img_len:]
    return train_list, val_list



def save_txt_index(train_list, val_list, save_path):
    save_meta_path = os.path.join(save_path, 'meta')
    mkdir(save_meta_path, is_remove=False)
    list_to_txt(os.path.join(save_meta_path, 'train.txt'), train_list)
    list_to_txt(os.path.join(save_meta_path, 'val.txt'), val_list)


def save_labelbee(width, height, index, result, dst):
    save_info = {
        "width": width,
        "height": height,
        "valid": 'true',
        "rotate": 0,
        "step_1": {
            "toolName": "tagTool",
            "result": [
                {
                    "id": index,
                    "sourceID": "",
                    "result": result,
                }
            ]
        }
    }
    save_json(save_info, dst)


def create_symlink(stat_info, total_list, save_path):
    image_path = os.path.join(save_path, 'images')
    label_path = os.path.join(save_path, 'labels')
    mkdir([image_path, label_path], is_remove=True)
    
    logger.info("Starting create symlink >>>")
    for index in tqdm(total_list):
        total_info = stat_info['TOTAL_INFO'][index]
        src = total_info['path']
        name = total_info['name']
        width = total_info['width']
        height = total_info['height']
        result = total_info['result']
        dst_image = os.path.join(image_path, name)
        dst_label = os.path.join(label_path, name + '.json') 
        os.symlink(src, dst_image)
        save_labelbee(width, height, index, result, dst_label)
    logger.info("All done!")


def main():
    save_path = '/home/jack/Projects/openmmlab/mmclassification/data/0302_total_multi_label_dataset'
    json_path = os.path.join(save_path, 'meta/stat_info.json')
    stat_info = load_json(json_path)
    total_list = list(stat_info['TOTAL_INFO'].keys())
    train_list, val_list = split_train_val_list(total_list, split_ratio=0.8)
    
    
    save_txt_index(train_list, val_list, save_path)
    create_symlink(stat_info, total_list, save_path)


if __name__ == '__main__':
    main()
