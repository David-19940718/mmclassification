import os
import cv2
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from tools.custom_tools.utils import mkdir, save_json, load_json, save_csv, list_to_txt

ATTRIBUTES = ['types', 'colors']
CLASSES = [
    ['bus', 'car', 'suv', 'truck', 'van'],
    ['black', 'blue', 'coffee', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
]
TOTAL_CLASS = [item for sublist in CLASSES for item in sublist]
NUM_PER_PRE_CLASS = [0] + [len(c) for c in CLASSES[:-1]]


def get_inds(gt, pd, mode):
    """
    Args:
        gt: ground truth
        pd: predictions
        mode: multilabel or multitest
    """
    # load json
    pred_json = load_json(os.path.join(pd, 'predictions.json'))
    target_json = load_json(gt)
    num_tasks = len(ATTRIBUTES)
    num_classes = len(TOTAL_CLASS)
    class_to_id = dict(zip(TOTAL_CLASS, (i for i in range(num_classes))))

    pos_inds, tar_inds = [], []
    pd, gt = [], []  # saved to compute confusion matrix
    error_info = []

    for index, item in pred_json.items():
        # predictions
        pred_index = np.zeros(num_classes, dtype=bool)
        pred_label = item['pred_label']  # [5, 6]
        if mode == 'multitask':
            for i in range(num_tasks):
                pred_label[i] += NUM_PER_PRE_CLASS[i]
        pred_index[pred_label] = True
        pos_inds.append(pred_index)
        
        # ground truths
        target_index = np.zeros(num_classes)
        target_class = list(target_json['TOTAL_INFO'][index]['result'].values())  # ['gray', 'car']
        target_label = [class_to_id[c] for c in target_class]
        target_index[target_label] = 1
        tar_inds.append(target_index)

        # error info
        if set(pred_label) != set(target_label):
            error_info.append({
                'img': target_json['TOTAL_INFO'][index]['path'],
                'gt': sorted(list(target_json['TOTAL_INFO'][index]['result'].values())),
                'pd': [TOTAL_CLASS[i] for i in pred_label],
                'score': item['pred_score'],
            })

        # confusion matrix
        pred_label.sort()
        target_label.sort()
        pd += pred_label
        gt += target_label

    return np.array(pos_inds), np.array(tar_inds), pd, gt, error_info


def cacl_multilabel_confusion_matrix(
    gt, pd, save_path, show=False, save=False
):

    labels = [i for i in range(len(TOTAL_CLASS))]
    cm = confusion_matrix(gt, pd, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TOTAL_CLASS)
    display.plot(
        include_values=True,   
        cmap="viridis",   
        ax=None,
        xticks_rotation="horizontal",
        values_format="d"
    )
    if save: 
        plt.tick_params(axis='x', labelsize=7)
        plt.tick_params(axis='y', labelsize=7)  
        plt.xticks(rotation=-20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confuse_matrix.jpg'))
    if show:
        plt.show()

    reports = classification_report(gt, pd, labels=labels, target_names=TOTAL_CLASS)
    logger.info("The confusion matrix is below:")
    print(reports)

    
def cacl_multilabel_performance(pred, target, save_path):
    """
    Example:
        pred (N, C) = [[True, False, False, False, Flase, True], [...], ...]
        target (N, C) = [[1, 0, 0, 1, 0, 0], [...], ...]
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    # Calculate basic metrics
    eps = np.finfo(np.float32).eps
    tp = (pred * target) == 1
    fp = (pred * (1 - target)) == 1
    fn = ((1 - pred) * target) == 1
    tn = ((1 - pred) * (1 - target)) == 1
    
    # Calculate each class score
    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    accuracy_class = np.maximum(tp.sum(axis=0) + tn.sum(axis=0), eps) / \
        np.maximum(tp.sum(axis=0) + tn.sum(axis=0) \
             + fp.sum(axis=0) + fn.sum(axis=0), eps
        )
    f1_class = 2 * precision_class * recall_class / \
        np.maximum(precision_class + recall_class, eps)

    # Svae metrics per class to csv file
    info = {
        "columns": TOTAL_CLASS,
        "index": ["accuracy", "precision", "recall", "f1"],
        "data": [
            accuracy_class * 100,
            precision_class * 100,
            recall_class * 100,
            f1_class * 100,   
        ]
    }
    save_csv(info, os.path.join(save_path, 'report.csv'))

    # Calculate per-class average score
    CA = accuracy_class.mean() * 100.0
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)

    # Calculate per-attribute average score
    ATTR_SCORE = {attr: dict() for attr in ATTRIBUTES}
    cnt = 0
    for i, c in enumerate(CLASSES):
        ATTR_SCORE[ATTRIBUTES[i]]['accuracy'] = \
            accuracy_class[cnt: len(c)].mean() * 100
        ATTR_SCORE[ATTRIBUTES[i]]['recall'] = \
            recall_class[cnt: len(c)].mean() * 100
        ATTR_SCORE[ATTRIBUTES[i]]['precision'] = \
            precision_class[cnt: len(c)].mean() * 100
        ATTR_SCORE[ATTRIBUTES[i]]['f1'] = \
            f1_class[cnt: len(c)].mean() * 100
        cnt += len(c)

    # Svae metrics per attribute to json file
    metrics = {
        "CLS_SCORE": dict(CA=CA, CP=CP, CR=CR, CF1=CF1),
        "ATTR_SCORE": ATTR_SCORE
    }
    save_json(metrics, os.path.join(save_path, 'report.json'))


def save_error(error_info, save_path, save_image=False, save_label=False):
    error_path = os.path.join(save_path, 'errors')
    error_list = []
    mkdir(error_path, is_remove=True)
    logger.info('>>>Starting save error images...')
    for info in tqdm(error_info):
        src = info['img']
        src_dst = info['img'] + '.json'
        lbl_dst = os.path.join(error_path, os.path.split(src)[-1]+'.json')
        lbl_info = load_json(src_dst)

        error_list.append(src)
        img = cv2.imread(src)
        dst = os.path.join(error_path, os.path.split(src)[-1])
        if save_image:
            for i in range(len(ATTRIBUTES)):
                label = "{}: {}={:.2f}%".format(
                    ATTRIBUTES[i], info['pd'][i], info['score'][i]
                )
                lbl_info['step_1']['result'][0]['result'][ATTRIBUTES[i]] = info['pd'][i] 
                cv2.putText(
                    img, label, (10, (i * 30) + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
            label = "GT: {}".format(info['gt'])
            cv2.putText(
                img, label, (10, ((i + 1) * 30) + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )
        cv2.imwrite(dst, img)
        if save_label:
            save_json(lbl_info, lbl_dst)

    list_to_txt(
        os.path.join(save_path, 'error_list.txt'), error_list, mode='w'
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('GT', help='Ground Truth.')
    parser.add_argument('PD', help='PreDiction.')
    parser.add_argument('--save-image', action='store_true', 
                        help='Saving error images.')
    parser.add_argument('--save-label', action='store_true', 
                        help='Saving error labels.')
    parser.add_argument('--mode', default=None, 
                        help='multilabel or multitask')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.mode in ['multilabel', 'multitask'], \
        f"Mode must be in multilabel or multitask, but got {args.mode}."

    pred, target, pd, gt, error_info = get_inds(args.GT, args.PD, args.mode)

    cacl_multilabel_performance(
        pred, target, args.PD
    )

    cacl_multilabel_confusion_matrix(
        gt, pd, args.PD, show=False, save=True,
    )

    if args.save_image or args.save_label: 
        save_error(
            error_info, 
            args.PD, 
            save_image=args.save_image, 
            save_label=args.save_label,
        )


if __name__ == '__main__':
    main()


