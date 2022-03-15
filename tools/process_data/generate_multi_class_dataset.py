import os
import json
import shutil
import random
from loguru import logger


"""
dataset_dir folder
|-- class1
    |-- 00001.jpg
    |-- ...
|-- class2
    |-- 00001.jpg
    |-- ...
|-- class3
    |-- 00001.jpg
    |-- ...
|-- ...
"""


def mkdir(p, is_remove=False):
    """Create a folder
    Args:
        p: file path. 
        is_remove: whether delete the exist folder or not. Default [False].
    """
    paths = p if isinstance(p, list) else [p]
    for p in paths:
        if is_remove and os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)


def parse_cls_data(dataset_dir):
    results, class_id = {}, [i for i in range(len(os.listdir(dataset_dir)))]

    for root, dirs, _ in os.walk(dataset_dir):
        dirs.sort()
        class_id.sort()
        results['num_classes'] = len(dirs)
        results['classes'] = dirs
        results['name2id'] = dict(zip(dirs, class_id))
        results['id2name'] = dict(zip(class_id, dirs))
        results['num_per_class'] = {}
        for d in dirs:
            results['num_per_class'][d] = len(os.listdir(os.path.join(root, d)))
        logger.info(results)
        return results


def split_train_val_list(total_list):
    random.shuffle(total_list)
    img_len = int(split_ratio * len(total_list))
    train_list = total_list[:img_len]
    val_list = total_list[img_len:]
    return train_list, val_list


def create_symlink(file_list, src_dir, dst_dir, classes):
    mkdir(os.path.join(dst_dir, classes), is_remove=True)

    for filename in file_list:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, classes, filename)
        os.symlink(src, dst)


def divide_dataset():
    for classes in os.listdir(dataset_dir):
        logger.info(f"Current processing class is [{classes}]")
        current_dirs = os.path.join(dataset_dir, classes)
        current_list = os.listdir(current_dirs)
        
        train_list, val_list = split_train_val_list(current_list)
        create_symlink(train_list, current_dirs, train_dir, classes)
        create_symlink(val_list, current_dirs, val_dir, classes)

def create_txt(results):
    mkdir(meta_dir, is_remove=True)
    for mode in ('train', 'val'):
        with open(os.path.join(meta_dir, mode+'.txt'), 'w') as f:
            if mode == 'train':
                total_dir = train_dir
            elif mode == 'val':
                total_dir = val_dir
            for classes in os.listdir(total_dir):
                for filename in os.listdir(os.path.join(total_dir, classes)):
                    current_id = results['name2id'][classes]
                    item = f'{classes}/{filename} {current_id}'
                    f.write(item + "\n")
    
    with open(os.path.join(meta_dir, 'class_info.json'), 'w') as f:
        json.dump(results, f) 


def main():
    divide_dataset()
    create_txt(parse_cls_data(dataset_dir))


if __name__ == '__main__':
    kfold = 0
    num_splits = 5 
    
    root_dir = '/home/jack/Projects/openmmlab/mmclassification/data/QingHaiContestKFold'
    meta_dir = os.path.join(root_dir, 'meta')
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    dataset_dir = '/data/workspace_jack/vehicle_attribute_dataset/competition/QingHaiContest/train/dataset'

    split_ratio = 0.8  # train: val

    main()