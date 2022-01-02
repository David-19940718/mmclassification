#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification


# Setting specific gpu device index
CUDA_VISIBLE_DEVICES=0
mode=$1
if [ $mode = "train" ];then
    echo "Starting training..."
    CONFIG_FILE=${HOME_PATH}/configs/vgg/vgg16_8xb16_voc.py 
    WORK_DIR=${HOME_PATH}/work_dirs/train/vgg16_8xb16_voc
    python ${HOME_PATH}/tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}
elif [ $mode = "test" ];then
    echo "Starting testing..."
    CONFIG_FILE=${HOME_PATH}/configs/vgg/vgg16_8xb16_voc.py 
    CHECKPOINT_FILE=${HOME_PATH}/work_dirs/train/vgg16_8xb16_voc/latest.pth
    OUT=${HOME_PATH}/work_dirs/test/vgg16_8xb16_voc/test.json
    # multi-label image classification
    python ${HOME_PATH}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUT} --metrics mAP CP CR CF1 OP OR OF1 --metric-options thr=0.5 k=2
else
    echo "Starting inference..."
    IMAGE_FILE=${HOME_PATH}/demo/cat-dog.png
    CONFIG_FILE=${HOME_PATH}/configs/vgg/vgg16_8xb16_voc.py
    CHECKPOINT_FILE=${HOME_PATH}/work_dirs/train/vgg16_8xb16_voc/latest.pth
    python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE}
fi

# sh start.sh train/test/demo
# define save best results, refs: https://github.com/open-mmlab/mmclassification/issues/541
