import os
import shutil
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot


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


def list_to_csv(p, info):
    """save *.csv file.
    Args:
        p: *.csv file path.
        info: saved infomation
            info = {
                "fileds": ['', '', ...], # title
                "rows": [[], [], ...]    # data
            }
    Returns:
        None
    """
    import csv
    with open(p, 'w') as f:
        write = csv.writer(f)
        write.writerow(info["fileds"])
        write.writerows(info["rows"])


def sort_key(s):
    import re
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def write_classes_info_to_pth(ckpt_file, classes=None):
    import torch
    checkpoint = torch.load(ckpt_file)
    checkpoint['meta']['CLASSES'] = classes
    torch.save(checkpoint, ckpt_file)
    logger.info("Successful to save class info into checkpoint.")


def create_preliminary_result(model, img, output=None, show=False, save=False):
    """
    result = {
        'pred_label': 1, 
        'pred_score': 0.663463294506073, 
        'pred_class': 'suv'
    }
    """
    if os.path.isfile(img):
        # test a single image
        result = inference_model(model, img)
        if show:
            # show the results
            show_result_pyplot(model, img, result)
    else:
        if output: mkdir(output, is_remove=True)
        fileds = ["id", "type"]
        rows = []
        imgs = os.listdir(img)
        imgs.sort(key=sort_key)
        bar = tqdm(imgs) if save else imgs
        for filename in bar:
            src = os.path.join(img, filename)
            result = inference_model(model, src)
            rows.append([filename, result["pred_class"]])
            if not save:
                logger.info(f"{filename} result -> {result}.")
        if save and output:
            list_to_csv(
                os.path.join(output, 'result.csv'), 
                dict(fileds=fileds, rows=rows),
            )
            logger.info(
                f"result.csv successful saved to {os.path.join(output, 'result.csv')}"
            )


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="/data/workspace_jack/vehicle_attribute_dataset/competition/QingHaiContest/testA",
                        help='Image file')
    parser.add_argument('--config', default="/home/jack/Projects/openmmlab/mmclassification/configs/custom/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314.py",
                        help='Config file')
    parser.add_argument('--checkpoint', default="/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/best_accuracy_top-1_epoch_406.pth",
                        help='Checkpoint file')
    parser.add_argument('--out', default='/data/workspace_jack/vehicle_attribute_dataset/competition/QingHaiContest/results/testA/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314_tta', 
                        help='Output path')
    parser.add_argument(
        '--device', default='cuda:6', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    classes = ["car", "suv", "truck", "van"]
    write_classes_info_to_pth(args.checkpoint, classes)
    model = init_model(args.config, args.checkpoint, device=args.device)

    create_preliminary_result(
        model,
        args.img,
        args.out,
        show=False,
        save=True,
    )

if __name__ == '__main__':
    main()




