import argparse
from ast import arg
from webbrowser import get
import torch.distributed as dist
import timm
import deepspeed
import torch
import benchmark.fairscale as fs
import benchmark.deepspeed as ds
from benchmark.configs.common import *
from benchmark.utils import get_time, parse_args
from benchmark.configs.config import Config


def main():
    args = parse_args()

    if args.type == 'ds':
        deepspeed.init_distributed()
    else:
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(dist.get_rank())

    if dist.get_rank() == 0:
        print('initialized torch distributed')
        print(vars(args))

    model = timm.models.vision_transformer.vit_large_patch16_224().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    if args.type == 'fs':
        model, optimizer, criterion = fs.init(model, criterion, args.stage)
        run_iter_func = fs.run_iter
    elif args.type == 'ds':
        model = ds.init(model, args)
        optimizer = None
        run_iter_func = ds.run_iter

    for _ in range(WARMUP):
        run_iter_func(model, optimizer, criterion)

    start = get_time()
    for _ in range(TEST_ITERS):
        run_iter_func(model, optimizer, criterion)
    end = get_time()

    print(
        f'average time: {(end - start) / TEST_ITERS}, ' +
        f'max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3}'
        + f'max memory cached: {torch.cuda.max_memory_cached() / 1024 ** 3}')


if __name__ == '__main__':
    main()