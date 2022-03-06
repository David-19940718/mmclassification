# Copyright (c) Jack.Wang. All rights reserved.
from argparse import ArgumentParser
from re import DEBUG

from mmcls.apis import inference_model, init_model

import os
import json
from tqdm import tqdm
from loguru import logger


def print_and_exit(msg):
    logger.debug(msg)
    os._exit(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('input', help='Image input path')
    parser.add_argument('output', help='Image output path')
    parser.add_argument(
        '--device', default='cuda:4', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    correct_class = os.path.basename(args.input)

    total_list = [os.path.join(args.input, p) for p in os.listdir(args.input)]

    error_list = {}

    items = total_list if DEBUG else tqdm(total_list)
    for item in items:
        result = inference_model(model, item)

        if not DEBUG:
            logger.info(item)
            logger.info(result)


        pred_class = result['pred_class']

        if DEBUG and pred_class != correct_class:
            logger.debug(f'predicted class is {result}, but correct class is {correct_class}.')
            logger.debug(item)
            error_list[item] = pred_class
    
    if not os.path.exists(os.path.split(args.output)[0]):
        os.makedirs(os.path.split(args.output)[0])
    
    if DEBUG:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(error_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Successful save to {args.output}")

if __name__ == '__main__':
    DEBUG = False
    main()
