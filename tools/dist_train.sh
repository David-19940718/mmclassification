#!/usr/bin/env bash
# /home/jack/Projects/openmmlab/mmclassification/configs/custom/swin_large_384_adamw_bs64_voc.py
PORT=${PORT:-29500}
CONFIG=$1
GPUS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
