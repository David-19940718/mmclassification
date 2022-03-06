import os
import shutil
from tqdm import tqdm

from tools.custom_tools.utils import load_json


def update_v1():
    # 将new数据集更新至old数据集
    old_root_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/dataset_2022-02-23'
    old_imgs_path = os.path.join(old_root_path, 'images')
    old_lbls_path = os.path.join(old_root_path, 'labels')
    new_root_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/standford_2022-02-23'
    new_imgs_path = os.path.join(new_root_path, 'images')
    new_lbls_path = os.path.join(new_root_path, 'labels')


    for img_name in tqdm(os.listdir(new_imgs_path)):
        lbl_name = img_name + '.json'
        src_img = os.path.join(new_imgs_path, img_name)
        dst_img = os.path.join(old_imgs_path, img_name)
        src_lbl = os.path.join(new_lbls_path, lbl_name)
        dst_lbl = os.path.join(old_lbls_path, lbl_name)
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(src_lbl, dst_lbl)


def update_v2():
    reviwe_lbl_path = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/eval/resnet50_sgd_multi_label_0227/errors'
    origin_lbl_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/check_2022-02-27'

    for filename in tqdm(os.listdir(reviwe_lbl_path)):
        lbl_file = os.path.join(reviwe_lbl_path, filename)
        lbl_info = load_json(lbl_file)
        result = lbl_info['step_1']['result'][0]['result']

        update_flag = True
        for a, c in result.items():
            if c not in CLASSES[ATTRIBUTES.index(a)]:
                update_flag = False
                os.remove(lbl_file)
                os.remove(os.path.join(origin_lbl_path, filename))
                os.remove(os.path.join(origin_lbl_path, filename.replace(".json", "")))
                break
        if update_flag:
            shutil.copyfile(lbl_file, os.path.join(origin_lbl_path, filename))
            os.remove(lbl_file)


if __name__ == '__main__':
    ATTRIBUTES = ['types', 'colors']
    CLASSES = [
        ['car', 'suv', 'truck', 'van'],
        ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
    ]
    update_v2()

