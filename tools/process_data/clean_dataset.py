import os
import cv2
import json
import shutil
from glob import glob
from tqdm import tqdm

ATTRIBUTES = ['types', 'colors']
CLASSES = [
    ['car', 'suv', 'truck', 'van'],
    ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow', 'others']
]
CLS_TO_ATTR = dict(zip(ATTRIBUTES, CLASSES))

def load_json(p):
    """load *.json file.
    """
    import json
    with open(p, 'r') as f:
        info = f.read()
    return json.loads(info)


def save_json(info, p):
    """save *.json file.
    Args:
        p: *.json file path.
        info: saved infomation
    Returns:
        None
    """
    save_info = json.dumps(info)
    with open(p, 'w') as f:
        f.write(save_info)


def mkdir(p, is_remove=False):
    """Create a folder
    Args:
        p: file path. 
        is_remove: whether delete the exist folder or not. Default [False].
    """
    paths = p if isinstance(p, list) else [p]
    for p in paths:
        if is_remove and os.path.exists(p):
            flag = input('Current dir is exist, whether remove or not? Please input [yes] or [no]:')
            if flag == 'yes':
                shutil.rmtree(p)
            else:
                continue
        os.makedirs(p, exist_ok=True)


def clean_dataset_v1(p):
    """Directory
    ROOT
        000000.jpg
        000000.jpg.json
        000001.jpg
        000001.jpg.json
        ...    
    """
    ignore_path = os.path.join(os.path.split(p)[0], 'ignore')
    others_path = os.path.join(os.path.split(p)[0], 'others')
    mkdir([ignore_path, others_path], is_remove=False)

    label_file = glob(os.path.join(p, '*.json'))
    for lbl_path in tqdm(label_file):
        info = load_json(lbl_path)
        result = info['step_1']['result'][0]['result']
        ignore_flag, others_flag = False, False
        for attr, clas in result.items():
            if clas not in CLS_TO_ATTR[attr]:
                ignore_flag = True
                break
            if clas == 'others': others_flag = True
        img_path = lbl_path.replace('.json', '')
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(lbl_path)
        if ignore_flag:
            shutil.move(img_path, os.path.join(ignore_path, img_name))
            shutil.move(lbl_path, os.path.join(ignore_path, lbl_name))
            continue       
        if others_flag:
            shutil.move(img_path, os.path.join(others_path, img_name))
            shutil.move(lbl_path, os.path.join(others_path, lbl_name))



def main():
    path = '/data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/check_2022-02-07'
    clean_dataset_v1(path)



if __name__ == '__main__':
    main()

