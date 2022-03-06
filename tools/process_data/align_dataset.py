import os
import shutil
from glob import glob


def count(paths):
    cnt = 0 
    for p in os.listdir(paths):
        cnt += len(os.listdir(os.path.join(paths, p)))
    return cnt

def create(paths):
    dic = dict()
    for cls in os.listdir(paths):
        path = os.path.join(paths, cls)
        for filename in os.listdir(path):
            dic[filename] = os.path.join(paths, cls, filename)
    return dic


if __name__ == '__main__':
    color_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/checks_2022-02-18/color'
    types_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/checks_2022-02-18/type'
    save_path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/checks_2022-02-18/trash'
    color_info = create(color_path)
    types_info = create(types_path)
    
    for key, val in color_info.items():
        if key not in types_info:
            shutil.move(val, os.path.join(save_path, key))

    for key, val in types_info.items():
        if key not in color_info:
            shutil.move(val, os.path.join(save_path, key))

    assert count(color_path) == count(types_path)
