#!/bin/bash

run_file=$1
stage=$2

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_addr localhost \
    --master_port 29500 \
    $run_file \
    --type fs \
    --stage $2
