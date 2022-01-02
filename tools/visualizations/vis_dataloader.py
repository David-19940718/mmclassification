import os
import numpy as np

import mmcv
from mmcv import Config

from mmcls.apis import init_random_seed
from mmcls.datasets import build_dataloader, build_dataset


def run_datatloader(cfg):
    """
    Visualize the effect of data augmentation, meanwhile, you can also confirm whether the custom training samples are correct.
    Args:
        cfg: configuration
    Returns:
        None
    """
    # Build dataset
    assert cfg.mode in ['train', 'val', 'test'], "cfg.mode should be in ['train', 'val', 'test']."
    if cfg.mode == 'train':
        dataset = build_dataset(cfg.data.train)
    elif cfg.mode == 'val':
        dataset = build_dataset(cfg.data.val)
    elif cfg.mode == 'test':
        dataset = build_dataset(cfg.data.test)

    # Prepare data loaders
    data_loader = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=len([cfg.gpu_ids]),
            round_up=True,
            seed=cfg.seed)

    for index, data_batch in enumerate(data_loader):
        img_batch = data_batch['img']
        gt_label = data_batch['gt_label']
        img_meta = data_batch['img_metas'].data[0]
        
        if index == RUN_BATCHS:
            exit(0)

        for batch_i in range(len(img_batch)):
            img = img_batch[batch_i]
            gt = gt_label[batch_i]
            ori_filename = img_meta[batch_i]['ori_filename']

            class_name = HUST_Vehicle_Color_Dataset.CLASSES[gt]
            mean_value = np.array(cfg.img_norm_cfg['mean'])
            std_value = np.array(cfg.img_norm_cfg['std'])
            img_hwc = np.transpose(img.numpy(), [1, 2, 0])
            img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
            img_numpy_uint8 = np.array(img_numpy_float, np.uint8)

            if IMAGE_SAVE:
                mmcv.imwrite(img_numpy_uint8, os.path.join(cfg.work_dirs, f'{class_name}_{os.path.split(ori_filename)[-1]}'))

            if IMAGE_SHOW:
                mmcv.imshow(img_numpy_uint8, 'img', 0)


if __name__ == '__main__':
    RUN_BATCHS = 1
    IMAGE_SAVE = True
    IMAGE_SHOW = False
   
    # replace your own custon Dataset
    from mmcls.datasets import HUST_Vehicle_Color_Dataset
    cfg = Config.fromfile('/home/jack/Projects/openmmlab/mmclassification/configs/_base_/datasets/hust_vehicle_color.py')

    # modify hyper parameters
    cfg.gpu_ids = 1  # gpu device index
    cfg.seed = init_random_seed(None)  # initialize random seed
    cfg.mode = 'train'  # 'train'/'val'/'test'
    cfg.work_dirs = f'/home/jack/Projects/openmmlab/mmclassification/work_dirs/vis/hust_vehicle_color/{cfg.mode}'  # save dir

    run_datatloader(cfg)