import torch.distributed as dist
import timm
import torch
from benchmark.configs.common import *
from benchmark.utils import get_time, parse_args
from benchmark.configs.config import Config
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def run_iter(model, optimizer, criterion):
    img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda().half()
    label = torch.randint(0, NUM_CLASS, (BATCH_SIZE, )).cuda()

    optimizer.zero_grad()
    out = model(img)
    out = out.float()
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()

def main():
    args = parse_args()
    dist.init_process_group(backend='nccl')

    torch.cuda.set_device(dist.get_rank())

    if dist.get_rank() == 0:
        print('initialized torch distributed')
        print(vars(args))

    model = timm.models.vision_transformer.vit_large_patch16_224().cuda().half()
    model = DDP(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)

    for _ in range(WARMUP):
        run_iter(model, optimizer, criterion)

    start = get_time()
    for _ in range(TEST_ITERS):
        run_iter(model, optimizer, criterion)
    end = get_time()

    print(
        f'average time: {(end - start) / TEST_ITERS}, ' +
        f'max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3}'
        + f'max memory cached: {torch.cuda.max_memory_cached() / 1024 ** 3}')


if __name__ == '__main__':
    main()