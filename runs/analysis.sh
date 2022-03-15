#!/bin/bash
export PYTHONPATH="${HOME}/Projects/openmmlab/mmcv":${PYTHONPATH}
HOME_PATH=${HOME}/Projects/openmmlab/mmclassification

mode=$1
if [ $mode = "res" ];then
    echo "Starting analysis results..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/qinghai_contest/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313.py
    RESULT_FILE=${HOME_PATH}/work_dirs/debug/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313/predictions.json
    OUTPUT_FILE=${HOME_PATH}/work_dirs/debug/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313/analysis_result
    python ${HOME_PATH}/tools/analysis_tools/analyze_results.py ${CONFIG_FILE} ${RESULT_FILE} --out-dir ${OUTPUT_FILE}
elif [ $mode = "flop" ];then
    echo "Starting get FLOPs..."
    CONFIG_FILE=${HOME_PATH}/configs/custom/qinghai_contest/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313.py
    python ${HOME_PATH}/tools/analysis_tools/get_flops.py ${CONFIG_FILE} --shape 224 224
elif [ $mode = "log" ];then
    echo "Starting analysis logs..."
    RESULT_FILE=${HOME_PATH}/work_dirs/debug/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313/predictions.json
    OUTPUT_FILE=${HOME_PATH}/work_dirs/debug/resnet18_pretrain_convnext_tricks_mixup_adamw_multi_class_220313/results.jpg
    # 绘制某日志文件对应的损失曲线图
    python tools/analysis_tools/analysis_logs.py plot_curve ${RESULT_FILE} --keys loss --legend loss
    # 绘制某日志文件对应的top-1准确率曲线图，并将曲线图导出为results.jpg 文件
    # python tools/analysis_tools/analysis_logs.py plot_curve ${RESULT_FILE} --keys accuracy_top-1 --legend top1 --out ${OUTPUT_FILE}
    # 在同一图像内绘制两份日志文件对应的top-1 准确率曲线图
    # python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys accuracy_top-1 --legend run1 run2

    
else
    echo "Done."
fi


