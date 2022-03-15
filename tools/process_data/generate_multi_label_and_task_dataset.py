import os
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


def create_train_symlink(stat_info, total_list, save_path):
    image_path = os.path.join(save_path, 'images')
    label_path = os.path.join(save_path, 'labels')
    mkdir([image_path, label_path], is_remove=True)
    
    logger.info("Starting create training dataset symlink >>>")
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


def create_test_dataset(save_path, test_path, stat_info):
    if not os.path.exists(os.path.join(save_path, "images")):
        os.makedirs(os.path.join(save_path, "images"))
    if not os.path.exists(os.path.join(save_path, "labels")):
        os.makedirs(os.path.join(save_path, "labels"))
    if not os.path.exists(os.path.join(save_path, "meta")):
        os.makedirs(os.path.join(save_path, "meta"))

    txt_path = os.path.join(save_path, 'meta/test.txt')
    json_path = os.path.join(save_path, 'meta/test_info.json')
    json_info = dict(BASIC_INFO=dict(), TOTAL_INFO=dict(), CLASS_INFO=dict())
    test_list, class_info = [], {}
    logger.info("Starting create testing dataset symlink >>>")
    for filename in tqdm(os.listdir(test_path)):
        index = os.path.splitext(filename)[0]
        src = os.path.join(test_path, filename)
        if filename.endswith('.json'):
            info = load_json(src)
            result = info["step_1"]["result"][0]["result"]
            json_info["TOTAL_INFO"][os.path.splitext(index)[0]] = {
                "height": info["height"],
                "width": info["width"],
                "name": filename.replace(".json", ""),
                "path": src.replace(".json", ""),
                "result": result
            }
            # caclulate info
            for a, c in result.items():
                if a not in class_info:
                    class_info[a] = {c: 1}
                else:
                    if c not in class_info[a]:
                        class_info[a][c] = 1
                    else:
                        class_info[a][c] += 1

            dst = os.path.join(save_path, "labels", filename)
        elif filename.endswith('.jpg'):
            test_list.append(index)
            dst = os.path.join(save_path, "images", filename)
        else:
            logger.debug(f"Please check this file {src}")
            exit(0)
        os.symlink(src, dst)
    
    # update test info
    basic_info = dict(
        NUM_ATTRIBUTES=len(class_info.keys()),
        NUM_CLASSES=dict()
    )
    for a in class_info.keys():
        basic_info["NUM_CLASSES"][a] = len(class_info[a].keys())
    json_info["BASIC_INFO"] = basic_info
    json_info["CLASS_INFO"] = class_info
    
    list_to_txt(txt_path, test_list)
    save_json(json_info, json_path)
    logger.info(f"Successful saved to {json_path}")


def main():
    save_path = '/home/jack/Projects/openmmlab/mmclassification/data/0310_train_standford_compcars_bus_test_benchmarkv1'
    test_path = '/data/workspace_jack/vehicle_attribute_dataset/test/benchmark_v1'

    # create train/val dataset
    json_path = os.path.join(save_path, 'meta/stat_info.json')
    stat_info = load_json(json_path)
    total_list = list(stat_info['TOTAL_INFO'].keys())
    train_list, val_list = split_train_val_list(total_list, split_ratio=0.8)
    save_txt_index(train_list, val_list, save_path)
    create_train_symlink(stat_info, total_list, save_path)

    if test_path:
        create_test_dataset(save_path, test_path, stat_info)
    logger.info("All done!")

if __name__ == '__main__':
    main()
