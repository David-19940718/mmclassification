#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification

# CUDA_VISIBLE_DEVICES=2,3 sh runs/start.sh dist_train

mode=$1
if [ $mode = "dist_train" ];then
    echo "Starting distrubuted training..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/convnext_xlarge_adamw_pretrain_multi_label_0303.py
    GPUS=2
    bash ${HOME_PATH}/tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
fi