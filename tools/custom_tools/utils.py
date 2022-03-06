import os
import cv2
import json
import shutil

import torch
import pandas as pd


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


def save_csv(info, p):
    """
    Args:
        info: {
            "data": [[], [], [], ...]
            "index": columns label
            "columns": rol label
        }
        p: save path
    """
    assert p.endswith(".csv"), \
        f"Excepted input '*.csv', instead got {p}" 
    df = pd.DataFrame(
        data=info["data"],
        index=info["index"],
        columns=info["columns"],
    )
    df.to_csv(p)


def mkdir(p, is_remove=False):
    """Create a folder
    Args:
        p: file path. 
        is_remove: whether delete the exist folder or not. Default [False].
    """
    paths = p if isinstance(p, list) else [p]
    for p in paths:
        if is_remove and os.path.exists(p):
            flag = input(f'Current dir {p} is exist, whether remove or not? Please input [yes] or [no]:')
            if flag == 'yes':
                shutil.rmtree(p)
            else:
                continue
        os.makedirs(p, exist_ok=True)


def txt_to_list(p, sep=' '):
    '''extract a list object from the *.txt file
    Args:
        p: *.txt file path.
        sep: Separators. [Default: ' ']
    Returns:
        A list object. [*, *, *, ...]
    '''
    with open(p) as f:
        info = f.readlines()
    res = [x.strip() for x in info]
    return res


def list_to_txt(p, lst, mode='w'):
    '''write a list object into txt file
    Args:
        p: *.txt file path
        lst: a list object
    Returns:
        None
    '''
    with open(p, mode) as f:
        for item in lst:
            item = " ".join(str(i) for i in item) if isinstance(item, list) else item
            f.write(item + "\n")


def save_labelbee(item, result):
    img = cv2.imread(item)
    index = os.path.splitext(os.path.basename(item))[0]
    height, width, _ = img.shape
    info = {
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
    save_json(info, item+'.json')


def move_error(src, dst):
    mkdir(dst, is_remove=False)
    load_info = load_json(src)
    for info in load_info.keys():
        file_name = os.path.split(info)[-1]
        save_name = os.path.join(dst, file_name)
        try:
            shutil.move(info, save_name)
        except FileNotFoundError:
            continue

    pass

def write_info_to_pth():
    pth_path = '/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_multi_label_0227/best_mAP_epoch_25.pth'
    checkpoint = torch.load(pth_path)
    checkpoint['meta']['CLASSES'] = (
        'car',
        'suv',
        'truck',
        'van',
        'black',
        'blue',
        'coffee',
        'gray',
        'green',
        'orange',
        'red',
        'white',
        'yellow',
    )
    print(checkpoint['meta']['CLASSES'])
    torch.save(checkpoint, pth_path)


if __name__ == '__main__':
    write_info_to_pth()