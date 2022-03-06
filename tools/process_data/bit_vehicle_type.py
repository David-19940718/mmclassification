import os
import cv2
import shutil

from scipy import io
from tqdm import tqdm
from loguru import logger

'''
name:       The filename of the image.
height:     The height of the image.
wideth:     The width of the image.
vehicles:   This field is a struct array with the size of 1*nvehicles, 
                and each element describes a vehicle with five fileds: left, top, right, bottom, and category. 
nvehicles:  The number of the vehicles in the image.
'''


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


def crop_rect(src, dst, pts, img_show=False, img_save=False):
    """Crop an image
    Args:
        src: source image path or array
        dst: save image path
        pts: [left, top, right, bottom]
        img_show: show image
        img_save: save_image
    Return:
        None
    """
    if isinstance(src, str):
        src = cv2.imread(src)

    crop_img = src[pts[1]: pts[3], pts[0]: pts[2]]

    if img_save:
        cv2.imwrite(dst, crop_img)

    if img_show:
        cv2.imshow(os.path.split(dst)[-1], crop_img)

def get_info(info):
    file_name    = info[0][0][0]
    height       = info[0][1][0][0]
    width        = info[0][2][0][0]
    vehicles     = info[0][3][0][0]
    num_vehicles = info[0][4][0][0]
    return str(file_name), int(height), int(width), vehicles, int(num_vehicles)


def get_vehicle(vehicles):
    left, top, right, bottom, cls_name = vehicles
    left     = int(left[0][0])
    top      = int(top[0][0])
    right    = int(right[0][0])
    bottom   = int(bottom[0][0])
    cls_name = str(cls_name[0])
    return left, top, right, bottom, cls_name


def process(VehicleInfo, class_to_id, root_dir, save_dir):
    '''e.g.
    ['vehicle_0001963.jpg']
    [[1080]]
    [[1920]]
    [[(array([[447]], dtype=uint16), array([[447]], dtype=uint16), array([[876]], dtype=uint16), array([[968]], dtype=uint16), array(['Sedan'], dtype='<U5'))
    (array([[1102]], dtype=uint16), array([[110]], dtype=uint8), array([[1394]], dtype=uint16), array([[432]], dtype=uint16), array(['Sedan'], dtype='<U5'))]]
    [[2]]    
    '''
    # {'Bus': 0, 'Microbus': 0, 'Minivan': 0, 'Sedan': 0, 'SUV': 0, 'Truck': 0}
    class_counts = dict(zip(list(class_to_id.keys()), [0 for _ in range(len(class_to_id.keys()))]))

    for info in tqdm(VehicleInfo):
        file_name, height, width, vehicles, num_vehicles = get_info(info)
        for n in range(num_vehicles):
            left, top, right, bottom, cls_name = get_vehicle(vehicles)
            src = os.path.join(root_dir, 'images', file_name)
            save_name = cls_name+'_height_'+str(bottom-top)+'_width_'+str(right-left)+'_'+str(class_counts[cls_name]).zfill(5)+'.jpg'
            dst = os.path.join(save_dir[class_to_id[cls_name]], save_name)
            class_counts[cls_name] += 1
            pts = [left, top, right, bottom]
            crop_rect(src, dst, pts, img_show=False, img_save=True)
    logger.info(f'total_info={class_counts}')


def main():
    ids = [i for i in range(6)]
    cls = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']
    class_to_id = dict(zip(cls, ids))

    root_dir = '/data/workspace_jack/vehicle_attribute_dataset/source/BIT_Vehicle_TYPE'
    save_dir = [os.path.join(root_dir, 'dataset', name) for name in cls]
    mkdir(save_dir, is_remove=True)

    mat = io.loadmat('/data/workspace_jack/vehicle_attribute_dataset/source/BIT_Vehicle_TYPE/VehicleInfo.mat') 
    logger.info(f"header={mat['__header__']}")
    logger.info(f"version={mat['__version__']}")
    VehicleInfo = mat['VehicleInfo']
    process(VehicleInfo, class_to_id, root_dir, save_dir)


if __name__ == '__main__':
    main()
