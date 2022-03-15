import os
import cv2
import json
from tqdm import tqdm
from loguru import logger

from tools.custom_tools.utils import save_json, load_json


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def init_attr_stat_info(ATTRIBUTES, CLASSES):
    """
    根据属性和类别列表初始化信息统计字典
    Args:
        ATTRIBUTES = ['colors', 'types']
        CLASSES = [
            ["red", "green", "blue"], 
            ["car", "suv", "truck"],
        ]
    Returns:
        stat_info = {
            'BASIC_INFO': {
                'NUM_ATTRIBUTES': len(ATTRIBUTES),
                'NUM_CLASSES': {'colors': 3, 'types': 3}
            },
            'CLASS_INFO' = {
                'colors': {
                    'red': 0,
                    'green': 0,
                    'blue': 0,
                },
                'types': {
                    'suv': 0,
                    'car': 0,
                    'van': 0,
                },
            },
            'TOTAL_INFO': None
        }
    """
    stat_info = {
        'BASIC_INFO': {
            'NUM_ATTRIBUTES': len(ATTRIBUTES),
            'NUM_CLASSES': {
                attr: len(CLASSES[i]) for i, attr in enumerate(ATTRIBUTES)
            }
        },
        'TOTAL_INFO': None
    }
    cls_info = []
    for i in range(len(ATTRIBUTES)):
        classes = CLASSES[i]
        numbers = [0 for _ in range(len(classes))]
        cls_info.append(dict(zip(classes, numbers)))
    stat_info.update({'CLASS_INFO': dict(zip(ATTRIBUTES, cls_info))})

    return stat_info


def update_info_v1(stat_info, root):
    """The dataset format is as follows.
    -| ROOT
        -| ATTR_A
            -| CLASS_A
            -| CLASS_B
            -| ...
        -| ATTR_B
            -| CLASS_A
            -| CLASS_B
            -| ...
        -| ...       
    """
    total_info = {}
    for attr in os.listdir(root):
        attr_path = os.path.join(root, attr)
        for clas in os.listdir(attr_path):
            clas_path = os.path.join(attr_path, clas)
            logger.info(f"Current process attr={attr} -> clas={clas}...")
            for file_name in tqdm(os.listdir(clas_path)):
                index = os.path.splitext(file_name)[0]
                curr_path = os.path.join(clas_path, file_name)
                if index not in total_info:
                    img = cv2.imread(curr_path)
                    h, w, _ = img.shape
                    total_info[index] = {
                        'height': h,
                        'width': w,
                        'name': file_name,
                        'path': curr_path,
                        'result': {attr: clas}
                    }
                else:
                    total_info[index]['result'].update({attr: clas})
                stat_info['CLASS_INFO'][attr][clas] += 1
    return total_info


def update_info_v2(stat_info, root):
    """The dataset format is as follows.
    -| ROOT
        -| images
            -| 000000.jpg
            -| 000001.jpg
            -| 000002.jpg
        -| labels [LabelBee format]
            -| 000001.jpg.json
            -| 000000.jpg.json
            -| 000002.jpg.json
        -| ...       
    """
    total_info = {}
    img_path = os.path.join(root, 'images')
    lbl_path = os.path.join(root, 'labels')
    for file_name in os.listdir(img_path):
        img_file = os.path.join(img_path, file_name)
        lbl_file = os.path.join(lbl_path, file_name+'.json')
        info = load_json(lbl_file)
        index = os.path.splitext(file_name)[0]
        result = info['step_1']['result'][0]['result']
        total_info[index] = {
            'height': info['height'],
            'width': info['width'],
            'name': file_name,
            'path': img_file,
            'result': result,
        }
        for attr, clas in result.items():
            stat_info['CLASS_INFO'][attr][clas] += 1
    return total_info


def update_info_v3(stat_info, root):
    """The dataset format is as follows.
    -| ROOT
        -| 000000.jpg
        -| 000001.jpg
        -| 000002.jpg
        -| 000001.jpg.json
        -| 000000.jpg.json
        -| 000002.jpg.json
        -| ...       
    """
    total_info = {}
    for file_name in tqdm(os.listdir(root)):
        if '.json' not in file_name: continue
        file_path = os.path.join(root, file_name)
        info = load_json(file_path)
        index = os.path.splitext(file_name.replace('.json', ''))[0]
        result = info['step_1']['result'][0]['result']
        total_info[index] = {
            'height': info['height'],
            'width': info['width'],
            'name': file_name.replace('.json', ''),
            'path': file_path.replace('.json', ''),
            'result': result,
        }
        for attr, clas in result.items():
            stat_info['CLASS_INFO'][attr][clas] += 1
    return total_info


def main():
    # Initialize
    ATTRIBUTES = ['types', 'colors']
    CLASSES = [
        ["bus", "car", "suv", "van", "truck"],
        ["black", "blue", "coffee", "gray", "green", "orange", "red", "white", "yellow"], 
    ]
    stat_info = init_attr_stat_info(ATTRIBUTES, CLASSES)
    
    # Process
    data1 = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/check_2022-02-27'
    logger.info(f'Current dataset -> {data1}')
    data1_dict = update_info_v3(stat_info, data1)

    data2 = '/data/workspace_jack/vehicle_attribute_dataset/source/CompCars/CompCars_1_20'
    logger.info(f'Current dataset -> {data2}')
    data2_dict = update_info_v3(stat_info, data2)

    data3 = '/data/workspace_jack/vehicle_attribute_dataset/source/MiniCars220308'
    logger.info(f'Current dataset -> {data3}')
    data3_dict = update_info_v3(stat_info, data3)

    data4 = '/data/workspace_jack/vehicle_attribute_dataset/source/Bus220308'
    logger.info(f'Current dataset -> {data4}')
    data4_dict = update_info_v3(stat_info, data4)

    # Merge datasets
    stat_info['TOTAL_INFO'] = merge_dicts(
        data1_dict, 
        data2_dict,
        # data3_dict,
        data4_dict,
    )

    # Saving results
    save_path = '/home/jack/Projects/openmmlab/mmclassification/data/0310_train_standford_compcars_bus_test_benchmarkv1/meta'
    os.makedirs(save_path, exist_ok=True)
    save_json(stat_info, os.path.join(save_path, 'stat_info.json'))

if __name__ == '__main__':
    main()
