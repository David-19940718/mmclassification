#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification

mode=$1
if [ $mode = "res" ];then
    echo "Starting analysis results..."


python tools/visualizations/vis_cam.py \
    /home/jack/Projects/openmmlab/mmclassification/data/HUST_Vehicle_Color/train/black/11131_11.jpg \
    /home/jack/Projects/openmmlab/mmclassification/configs/custom/resnet50_sgd_bs64_hust.py  \
    /home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_bs64_hust/best.pth \
    --save-path /home/jack/Projects/openmmlab/mmclassification/work_dirs/vis/vis_cam/HUST_Vehicle_Color/11131_11.jpg

    # --target-layers 'backbone.layer4.2' \
    # --method GradCAM \
    # --target-category 5 \