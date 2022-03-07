
# Copyright (c) Jack.Wang. All rights reserved.
import os
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

from mmcls.apis import init_model

from tools.custom_tools.utils import save_labelbee
from tools.custom_tools.inference import inference_multi_task_model


ATTRIBUTES = ['colors', 'types']
CLASSES = [
    ['black', 'blue', 'yellow'],
    ['bus', 'car', 'suv'],
]

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('input', default=None, help='Image input path')
    parser.add_argument('--DEBUG', default=False, help='Debug')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    items = [os.path.join(args.input, p) for p in os.listdir(args.input)]

    # predict
    if args.DEBUG:
        for item in items:
            try:
                _ = inference_multi_task_model(model, item, CLASSES)
            except AttributeError:
                logger.debug(item)
        logger.info("Everything is ok! You can reset the DEBUG False.")
    else:
        for item in tqdm(items):
            res = inference_multi_task_model(model, item, CLASSES, mode='tag')
            result = {ATTRIBUTES[i]: res['pred_class'][i] for i in range(len(ATTRIBUTES))}
            save_labelbee(item, result)


if __name__ == '__main__':
    main()
