#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29504}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3}
