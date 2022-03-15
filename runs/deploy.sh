#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification

# CUDA_VISIBLE_DEVICES=2 sh runs/deploy.sh onnx2trt
# CUDA_VISIBLE_DEVICES=3 sh runs/deploy.sh pth2onnx

mode=$1
if [ $mode = "pth2onnx" ];then
    echo "Starting convert *.pth to *.onnx"
    CONFIG=${HOME_PATH}/configs/custom/resnet50_sgd_multi_label_0310.py 
    CHECKPOINT=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/benchmark_v1/resnet50_sgd_multi_label_0310/latest.pth
    OUTPUT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/benchmark_v1/resnet50_sgd_multi_label_0310/latest.onnx
    python ${HOME_PATH}/tools/deployment/pytorch2onnx.py ${CONFIG} --checkpoint ${CHECKPOINT} --output-file ${OUTPUT_FILE} --shape 224 224 --opset-version 11 --verify --batch 16
elif [ $mode = "onnx2trt" ];then
    echo "Starting convert *.onnx to *.trt"
    MODEL=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_multi_label_0211_vanilla_settings/latest.onnx
    # MODEL=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_bs64_gender_pretrain_fronzen_stage_2/best_accuracy_top-1_epoch_35.onnx
    TRT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_multi_label_0211_vanilla_settings/latest.trt
    # TRT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_bs64_gender_pretrain_fronzen_stage_2/best_accuracy_top-1_epoch_35.trt
    MAX_BATCH_SIZE=1
    python ${HOME_PATH}/tools/deployment/onnx2tensorrt.py ${MODEL} --trt-file ${TRT_FILE} --fp16 --max-batch-size ${MAX_BATCH_SIZE} --verify
elif [ $mode = "jzp2onnx" ];then
    echo "Starting convert junzheng *.pth to *.onnx"
    CONFIG=${HOME_PATH}/configs/custom/jz_t40_resnet_multi_lable.py 
    CHECKPOINT=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/jz_t40_resnet_multi_lable_4bit/latest.pth
    OUTPUT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/jz_t40_resnet_multi_lable_4bit/latest.onnx
    python ${HOME_PATH}/tools/deployment/jz_pytorch2onnx.py ${CONFIG} --checkpoint ${CHECKPOINT} --output-file ${OUTPUT_FILE} --opset-version 9 --shape 224 224
else
    echo "Done."
fi