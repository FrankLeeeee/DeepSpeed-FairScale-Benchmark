#!/bin/bash

bash ./scripts/run_fairscale.sh run_benchmark.py 1 > ./fs_1.log
bash ./scripts/run_fairscale.sh run_benchmark.py 2 > ./fs_2.log
bash ./scripts/run_fairscale.sh run_benchmark.py 3 > ./fs_3.log

bash scripts/run_deepspeed.sh  run_benchmark.py benchmark/configs/deepspeed_stage1.json > ./ds_1.log
bash scripts/run_deepspeed.sh  run_benchmark.py benchmark/configs/deepspeed_stage2.json > ./ds_2.log
bash scripts/run_deepspeed.sh  run_benchmark.py benchmark/configs/deepspeed_stage3.json > ./ds_3.log