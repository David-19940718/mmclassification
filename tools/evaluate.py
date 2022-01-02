from argparse import ArgumentParser

from mmcv import Config
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.apis import inference_model, init_model, single_gpu_test

"""
python tools/evaluate.py --out_dir results/hust_results/evaluate --config configs/resnet/resnet50_8xb32_in1k.py --checkpoint results/hust_results/epoch_99.pth
"""

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', help='Image folder')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # initialize config
    cfg = Config.fromfile(args.config)
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        round_up=True)
    # evaluate metrics
    results = single_gpu_test(
        model=model,
        data_loader=test_dataloader,
        show=False,
        out_dir=args.out_dir,
    )
    print(results)


if __name__ == '__main__':
    main()
