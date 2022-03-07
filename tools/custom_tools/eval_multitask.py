# Copyright (c) Jack.Wang. All rights reserved.

import os
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

from tools.custom_tools.utils import mkdir, txt_to_list, save_json
from tools.custom_tools.inference import inference_multi_task_model

import mmcv
from mmcls.apis import init_model


CLASSES = [
    ['black', 'blue', 'yellow'],
    ['bus', 'car', 'suv']
]
ATTRIBUTES = ['colors', 'types']


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('output', default=None, help='Image output path')
    parser.add_argument('--flag', default='test', help='train/val/test/total')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # get predicted list
    predictions = dict()
    cfg = mmcv.Config.fromfile(args.config)

    # choose index you want to predict
    items = os.path.join(cfg.data.train['data_prefix'], 'images')
    if args.flag == 'train':
        retain_indexs = txt_to_list(cfg.data.train['ann_file'])
    elif args.flag == 'val':
        retain_indexs = txt_to_list(cfg.data.val['ann_file'])
    elif args.flag == 'test':
        retain_indexs = txt_to_list(cfg.data.test['ann_file'])
    elif args.flag == 'total':
        pass
    else:
        logger.debug('flag must be in [train, val, test, total].')

    for item in tqdm(os.listdir(items)):
        index = os.path.splitext(item)[0]
        # if the current index not in test dataset
        if args.flag != 'total' and index not in retain_indexs:
            continue
        predictions[index] = inference_multi_task_model(
            model, os.path.join(items, item), CLASSES
        )
    # save result
    if args.output:
        mkdir(os.path.split(args.output)[0], is_remove=False)
        save_json(predictions, args.output)


if __name__ == '__main__':

    main()
