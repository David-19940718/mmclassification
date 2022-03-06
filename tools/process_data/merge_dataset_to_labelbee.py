"""
根据LabelBee格式划分为对应的属性和标签
|-- src_path
    |-- Attribute_a
        |-- Class_1
            |-- 000000.jpg
            |-- 000001.jpg
        |-- Class_2
        |-- ...
    |-- ...
|-- dst_path
    |-- images
    |-- labels
"""

import os
import json
import shutil
import datetime
from tqdm import tqdm
from loguru import logger

from tools.custom_tools.utils import mkdir, load_json, save_json


TODAY_DATE = str(datetime.date.today())


def save_labelbee_json(src, dst, info):
    import cv2
    img = cv2.imread(src)
    height, width, _ = img.shape
    save_info = {
        "width": width,
        "height": height,
        "valid": "true",
        "rotate": 0,
        "step_1": {
            "toolName": "tagTool",
            "result": [
                {
                    "id": dst,
                    "sourceID": "",
                    "result": info,
                }
            ]
        }
    }
    save_json(save_info, dst)


def merge_v1():


    def update_labelbee_json(dst, info):
        save_info = load_json(dst)
        save_info["step_1"]["result"][0]["result"].update(info)
        save_json(save_info, dst)

    src_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/checks_2022-02-18'
    dst_path = os.path.join(os.path.split(src_path)[0], 'dataset_'+TODAY_DATE)
    dst_img_path = os.path.join(dst_path, 'images')
    dst_lbl_path = os.path.join(dst_path, 'labels')
    ignore_path = os.path.join(dst_path, 'ignores')
    logger.info(f'dst_img_path={dst_img_path}')
    logger.info(f'dst_lbl_path={dst_lbl_path}')
    logger.info(f'ignore_path={ignore_path}')
    mkdir([dst_img_path, dst_lbl_path, ignore_path], is_remove=True)

    # 将含有对应标签的文件过滤掉
    move_flag, move_list, move_target = False, set(), ['ignore']
    # 需要保留的属性
    att_target = ['colors', 'types']

    att_list = os.listdir(src_path)
    for att in att_list:
        if att not in att_target: continue
        att_path = os.path.join(src_path, att)
        for cls in os.listdir(att_path):
            cls_path = os.path.join(att_path, cls)
            logger.info(f'Current process: Attribute-{att} and Class-{cls}>>>')
            for file_name in tqdm(os.listdir(cls_path)):
                src = os.path.join(cls_path, file_name)
                save_img_path = os.path.join(dst_img_path, file_name)
                if att in move_target or cls in move_target:
                    move_list.add(file_name)
                if not os.path.isfile(save_img_path):
                    shutil.copyfile(src, save_img_path)
                save_lbl_path = os.path.join(dst_lbl_path, file_name+'.json')
                if os.path.isfile(save_lbl_path):
                    update_labelbee_json(save_lbl_path, {att: cls})
                else:
                    save_labelbee_json(src, save_lbl_path, {att: cls})

    if move_flag:
        logger.info(f">>>Remove ignore files")
        for file_name in os.listdir(dst_img_path):
            if file_name in move_list:
                src_img = os.path.join(dst_img_path, file_name)
                dst_img = os.path.join(ignore_path, file_name)
                src_lbl = os.path.join(dst_lbl_path, file_name + '.json')
                shutil.copyfile(src_img, dst_img)
                os.remove(src_img)
                os.remove(src_lbl)


def merge_v2():
    """
    Input:
        ROOT
            red-car
            red-suv
            ...
            black-car
            black-suv
            ...
    Output:
        ROOT
            000000.jpg
            000001.jpg.json
    """
    CLASSES = [
        ['bus', 'car', 'suv', 'truck', 'van'],
        ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
    ]
    root = '/data/workspace_jack/vehicle_attribute_dataset/custom'
    save = '/data/workspace_jack/vehicle_attribute_dataset/test/benchmark_v1'
    mkdir(save, is_remove=False)

    for c in CLASSES[1]:
        for t in CLASSES[0]:
            path = os.path.join(root, c+'-'+t)
            logger.info(f">>> Current process {c}-{t}...")
            for filename in tqdm(os.listdir(path)):
                src = os.path.join(path, filename)
                dst = os.path.join(save, filename)
                shutil.copyfile(src, dst)
                info = {"colors": c, "types": t}
                save_labelbee_json(src, dst+'.json', info)



if __name__ == '__main__':
    merge_v2()


