import os
import json
import shutil
import datetime
from tqdm import tqdm

from tools.custom_tools.utils import mkdir, load_json

TODAY_DATE = str(datetime.date.today())



def divide_v1():
    """
    根据LabelBee格式划分为对应的属性和标签
    |-- src_path
        |-- images
        |-- labels
    |-- dst_path
        |-- Attribute_a
            |-- Class_1
            |-- Class_2
            |-- ...
        |-- Attribute_b
        |-- ignore
        |-- ...
    """

    src_path = '/home/jack/Projects/yolov5/runs/detect/standford_cars_dataset/exp2/crops/car'
    img_path = os.path.join(src_path, 'images')
    lbl_path = os.path.join(src_path, 'labels')
    dst_path = os.path.join(src_path, 'checks_'+TODAY_DATE)
    ign_path = os.path.join(dst_path, 'ignore')
    mkdir(dst_path, is_remove=True)

    ign_list, att_list = set(), []

    for label_name in tqdm(os.listdir(lbl_path)):
        lbl_file = os.path.join(lbl_path, label_name)
        lbl_info = load_json(lbl_file)
        total_info = lbl_info['step_1']['result'][0]['result']
        if not att_list:
            att_list = list(total_info.keys())

        for att, cls in total_info.items():
            save_path = os.path.join(dst_path, att, cls)
            if cls == 'ignore':
                ign_list.add(image_name)
                continue
            if not os.path.exists(save_path): 
                mkdir(save_path, is_remove=False)
            image_name = os.path.splitext(label_name)[0]

    mkdir(ign_path, is_remove=True)
    for file_name in tqdm(os.listdir(img_path)):
        src = os.path.join(img_path, file_name)
        if file_name in ign_list:
            dst = os.path.join(ign_path, file_name)
            shutil.copyfile(src, dst)
            continue
        
        lbl_file = os.path.join(lbl_path, file_name + '.json')
        lbl_info = load_json(lbl_file)
        total_info = lbl_info['step_1']['result'][0]['result']
        for att, cls in total_info.items():
            save_path = os.path.join(dst_path, att, cls)
            dst = os.path.join(save_path, file_name)
            shutil.copyfile(src, dst)



def divide_v2():

    CLASSES = {
        'types': ['car', 'suv', 'truck', 'van'],
        'colors': ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
    }
    root_path = '/data/workspace_jack/vehicle_attribute_dataset/source/Hyper/hyper'
    save_path = '/data/workspace_jack/vehicle_attribute_dataset/source/Hyper/uncheck_hyper'
    mkdir(save_path, is_remove=True)
    for filename in tqdm(os.listdir(root_path)):
        if ".json" not in filename: continue    
        lbl_file = os.path.join(root_path, filename)
        lbl_info = load_json(lbl_file)
        result = lbl_info['step_1']['result'][0]['result']
        if result['colors'] not in CLASSES['colors']: continue
        if result['types'] not in CLASSES['types']: continue
        tmp = os.path.join(save_path, result['colors']+'-'+result['types'])
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        src = os.path.join(lbl_file.replace('.json', ''))
        dst = os.path.join(tmp, filename.replace('.json', ''))
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    divide_v2()








