# Copyright (c) Jack.Wang. All rights reserved.

import os
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

from tools.custom_tools.utils import mkdir, txt_to_list, save_json
from tools.custom_tools.inference import \
    inference_multi_label_model, inference_multi_task_model

import mmcv
from mmcls.apis import init_model


ATTRIBUTES = ['types', 'colors']
CLASSES = [
    ['bus', 'car', 'suv', 'truck', 'van'],
    ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
]


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('output', default=None, help='Image output path')
    parser.add_argument('--flag', nargs='+', type=str, help='--flag train val')
    parser.add_argument('--mode', default=None, 
                        help='multilabel or multitask')
    parser.add_argument('--device', default='cuda:0', 
                        help='Device used for inference')
    args = parser.parse_args()

    assert args.mode in ['multilabel', 'multitask'], \
        f"Mode must be in multilabel or multitask, but got {args.mode}."
    
    assert set(args.flag).issubset(['train', 'val', 'test']), \
        f"flag must be in ['train', 'val', 'test'], but got {args.flag}"

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # get predicted list
    predictions = dict()
    cfg = mmcv.Config.fromfile(args.config)

    # choose index you want to predict
    retain_indexs = []
    if 'train' in args.flag:
        retain_indexs += txt_to_list(cfg.data.train['ann_file'])
    if 'val' in args.flag:
        retain_indexs += txt_to_list(cfg.data.val['ann_file'])
    if 'test' in args.flag:
        retain_indexs += txt_to_list(cfg.data.test['ann_file'])

    # prediction
    items = os.path.join(cfg.data.train['data_prefix'], 'images')
    for item in tqdm(os.listdir(items)):
        index = os.path.splitext(item)[0]
        img = os.path.join(items, item)

        if index not in retain_indexs: continue

        if args.mode == 'multilabel':
            results = inference_multi_label_model(
                model, img, CLASSES
            )
        else:
            results = inference_multi_task_model(
                model, img, CLASSES
            )  
        predictions[index] = results

        if not args.output:
            logger.info(f"{img} -> results")

    # save result
    if args.output:
        mkdir(os.path.split(args.output)[0], is_remove=False)
        save_json(predictions, args.output)


if __name__ == '__main__':

    main()
