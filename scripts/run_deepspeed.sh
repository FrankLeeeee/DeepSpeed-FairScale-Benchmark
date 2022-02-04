#!/bin/bash

run_file=$1
config=$2

deepspeed $run_file \
        --type ds \
        --deepspeed \
        --deepspeed_config $config