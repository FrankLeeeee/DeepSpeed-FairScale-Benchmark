import argparse
import torch
import time
import deepspeed


def get_time():
    torch.cuda.synchronize()
    return time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['ds', 'fs'])
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--stage', type=int)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.stage:
        assert args.type == 'fs', 'argument stage is only for fairscale'
    return args
