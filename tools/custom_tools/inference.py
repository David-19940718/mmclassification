import torch
import numpy as np

from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose


def inference_multi_label_model(model, img, CLASSES, mode='eval'):
    """Inference image(s) with the multi-label classifier.
    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
        CLASSES： Total classes
    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label`, and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():

        scores = model(return_loss=False, **data)

        # 按属性输出
        topk_pred_score, topk_pred_index, topk_pred_class, cnt = [], [], [], 0
        for c in range(len(CLASSES)):
            score = scores[0][cnt: cnt+len(CLASSES[c])]
            pred_score = np.max(score)
            pred_index = np.argmax(score) + cnt
            if mode == 'eval':
                pred_label = model.CLASSES[pred_index]
            elif mode == 'tag':
                pred_label = model.CLASSES[pred_index] if pred_score > 0.5 else 'others'
            topk_pred_score.append(pred_score)
            topk_pred_index.append(pred_index)
            topk_pred_class.append(pred_label)
            cnt += len(CLASSES[c])
    
        '''多头输出
        topk_pred_score = heapq.nlargest(num_attrs, scores[0])
        topk_pred_index = heapq.nlargest(num_attrs, range(len(scores[0])), scores[0].take)
        topk_pred_index = [
            topk_pred_index[i] \
                if topk_pred_score[i] > 0.5 else len(model.CLASSES) \
                    for i in range(num_attrs)
        ]
        topk_pred_class = [
            model.CLASSES[index] if score > 0.5 else 'others' for index, score in zip(max_topk_pred_index, topk_pred_score)
        ]
        '''
    result = {
        'pred_score': np.array(topk_pred_score).tolist(),
        'pred_label': np.array(topk_pred_index).tolist(),
        'pred_class': topk_pred_class,
    }
    return result

