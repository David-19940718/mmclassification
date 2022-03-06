import os
import shutil
from tqdm import tqdm


def txt_to_list_v2(p, sep=' '):
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


def extract_from_origin():
    image_dir = '/data/workspace_jack/vehicle_attribute_dataset/source/PKU_VD_Brand_Color/tmp/PKU-VD/VD1/image'
    train_dir = '/data/workspace_jack/vehicle_attribute_dataset/source/PKU_VD_Brand_Color/tmp/PKU-VD/VD1/train_test/trainlist.txt'
    test_dir = '/data/workspace_jack/vehicle_attribute_dataset/source/PKU_VD_Brand_Color/tmp/PKU-VD/VD1/train_test/testlist.txt'

    image_list = os.listdir(image_dir)
    train_list = txt_to_list_v2(train_dir)
    test_list = txt_to_list_v2(test_dir)
    total_list = train_list + test_list

    files = []
    labels = []

    for _, val in enumerate(total_list):
        tmp = val.split(' ')
        files.append(tmp[0])
        labels.append(tmp[-1])

    counts = {
        str(i): 0 for i in range(11)
    }

    save_root = '/data/workspace_jack/vehicle_attribute_dataset/source/PKU_VD_Brand_Color/tmp/PKU-VD/VD1/check_pku_vd'
    save_path = [os.path.join(save_root, val) for val in counts.keys()]
    mkdir(save_path, is_remove=True)

    for filename in tqdm(image_list):
        basename = os.path.splitext(filename)[0]
        if basename in files:
            index = files.index(basename)
            label = labels[index]
            src = os.path.join(image_dir, filename)
            dst = os.path.join(save_path[int(label)], filename)
            if counts[label] < 100:
                shutil.copy(src, dst)
            counts[label] += 1


def main():
    extract_from_origin()


if __name__ == '__main__':
    main()



