#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification

# CUDA_VISIBLE_DEVICES=6,7 sh runs/start.sh dist_train
# CUDA_VISIBLE_DEVICES=7 sh runs/start.sh train
# CUDA_VISIBLE_DEVICES=3 sh runs/start.sh eval
# CUDA_VISIBLE_DEVICES=6 sh runs/start.sh tag
# CUDA_VISIBLE_DEVICES=7 sh runs/start.sh distill_train
# CUDA_VISIBLE_DEVICES=7 sh runs/start.sh distill_test
# CUDA_VISIBLE_DEVICES=7 sh runs/start.sh test
# sh runs/start.sh valid

mode=$1
if [ $mode = "dist_train" ];then
    echo "Starting distrubuted training..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314_kfold0.py
    GPUS=2
    bash ${HOME_PATH}/tools/dist_train.sh ${CONFIG_FILE} ${GPUS}

elif [ $mode = "train" ];then
    echo "Starting training..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314.py
    WORK_DIR=${HOME_PATH}/work_dirs/train/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314
    python ${HOME_PATH}/tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

elif [ $mode = "test" ];then
    echo "Starting testing..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314.py
    CHECKPOINT_FILE=${HOME_PATH}/work_dirs/train/qinghai_contest/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/best_accuracy_top-1_epoch_406.pth
    OUTPUT_FILE=${HOME_PATH}/work_dirs/debug/convnextLarge_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/predictions.json
    python ${HOME_PATH}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_FILE} --metrics accuracy recall precision f1_score --metric-options topk=1

elif [ $mode = "distill_train" ];then
    echo "Starting distill training..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/resnet18_pretrain_convnext_tricks_sgd_multi_class_220313.py
    WORK_DIR=${HOME_PATH}/work_dirs/train/resnet18_pretrain_convnext_tricks_sgd_multi_class_220313
    python ${HOME_PATH}/tools/distill_train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

elif [ $mode = "distill_test" ];then
    echo "Starting distill testing..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314.py
    CHECKPOINT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/best_mAP_epoch_17.pth
    OUTPUT_FILE=${HOME_PATH}/work_dirs/eval/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/predictions.json
    python ${HOME_PATH}/tools/distill_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${OUTPUT_FILE} --metrics mAP CP CR CF1

elif [ $mode = "eval" ];then
    echo "Starting evaluating..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314.py
    CHECKPOINT_FILE=/home/jack/Projects/openmmlab/mmclassification/work_dirs/train/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/best_mAP_epoch_17.pth
    OUTPUT_FILE=${HOME_PATH}/work_dirs/eval/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314/predictions.json
    python ${HOME_PATH}/tools/custom_tools/eval_multilabel_and_multitask.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_FILE} --flag test --mode multilabel

elif [ $mode = "valid" ];then
    echo "Starting validating..."
    GT=${HOME_PATH}/data/0308_train_standford_compcars_mini_bus_test_benchmarkv1/meta/test_info.json
    PD=${HOME_PATH}/work_dirs/eval/benchmark_v1/resnet50_pretrain_convnext_tricks_mixup_adamw_multi_class_220314
    python ${HOME_PATH}/tools/custom_tools/test_multilabel_and_multitask.py ${GT} ${PD} --mode multilabel 

elif [ $mode = "tag" ];then
    echo "Starting tagging..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/convnext_xlarge_adamw_pretrain_multi_label_0303.py
    CHECKPOINT_FILE=${HOME_PATH}/work_dirs/train/convnext_xlarge_adamw_pretrain_multi_label_0306/latest.pth
    INPUT_FILE=/data/workspace_jack/vehicle_attribute_dataset/source/CompCars/tmp
    python ${HOME_PATH}/tools/custom_tools/tag_multilabel.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${INPUT_FILE} --mode multilabel
fi

