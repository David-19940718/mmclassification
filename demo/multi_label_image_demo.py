# Copyright (c) Jack.Wang. All rights reserved.
import os
import cv2
import numpy as np
from loguru import logger
from argparse import ArgumentParser

import torch
from mmcls.apis import init_model
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose

from tools.custom_tools.utils import mkdir


def get_results(model, inputs, CLASSES) -> dict:
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(inputs, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=inputs), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=inputs)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    with torch.no_grad():
        scores = model(return_loss=False, **data)
        topk_pred_score, topk_pred_index, topk_pred_class, cnt = [], [], [], 0

        for c in range(len(CLASSES)):
            score = scores[0][cnt: cnt+len(CLASSES[c])]
            pred_score = np.max(score)
            pred_index = np.argmax(score) + cnt
            pred_label = model.CLASSES[pred_index] if pred_score > 0.5 else 'others'
            topk_pred_score.append(pred_score)
            topk_pred_index.append(pred_index)
            topk_pred_class.append(pred_label)
            cnt += len(CLASSES[c])
    results = {
        'topk_pred_score': topk_pred_score,
        'topk_pred_index': topk_pred_index,
        'topk_pred_class': topk_pred_class,
    }
    logger.info(f"Inference results for {inputs} as follow: \n{results}")

    return results


def save_image(inputs, output, results, ATTRIBUTES, is_save=False):
    img = cv2.imread(inputs) if isinstance(inputs, str) else inputs
    for i in range(len(ATTRIBUTES)):
        label = "{}: {}={:.2f}%".format(
            ATTRIBUTES[i], 
            results['topk_pred_class'][i], 
            results['topk_pred_score'][i] * 100,
        )
        cv2.putText(
            img, label, (10, (i * 30) + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )
    if is_save:
        dst = os.path.join(output, os.path.split(inputs)[-1])
        cv2.imwrite(dst, img)
        logger.info(f'Successful saved to {dst}')


def save_video(model, inputs, output, CLASSES, ATTRIBUTES):
    union_sizes = (224, 224)
    total_frame = 1
    dst = os.path.join(output, os.path.split(inputs)[-1]+'.mp4')
    video_writer = cv2.VideoWriter(
        dst, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        total_frame, 
        union_sizes,
    )
    img_array = []
    for filename in os.listdir(inputs):
        cur_file = os.path.join(inputs, filename)
        img = cv2.imread(cur_file)
        img = cv2.resize(img, union_sizes)
        results = get_results(model, cur_file, CLASSES)
        save_image(img, output, results, ATTRIBUTES, False)
        img_array.append(img)
        video_writer.write(img)
    for i in range(len(img_array)):
        video_writer.write(img_array[i])
    video_writer.release()
    logger.info(f'Successful saved to {dst}')


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('src', default=None, help='Image input path')
    parser.add_argument('dst', default=False, help='Debug')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    ATTRIBUTES = ['types', 'colors']
    CLASSES = [
        ['car', 'suv', 'truck', 'van'],
        ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
    ]

    mkdir(args.dst, is_remove=False)

    # build the model from a config file and a checkpoint file.
    model = init_model(args.config, args.checkpoint, args.device)

    # save image
    if os.path.isfile(args.src):
        # inference image and obtain results.
        results = get_results(model, args.src, CLASSES)
        # plot result on image and saved.
        save_image(args.src, args.dst, results, ATTRIBUTES, True)
    elif os.path.isdir(args.src):
        save_video(model, args.src, args.dst, CLASSES, ATTRIBUTES)


if __name__ == '__main__':
    main()
