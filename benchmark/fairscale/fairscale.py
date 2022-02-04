import torch
from fairscale.optim.oss import OSS
from ..configs.common import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from fairscale.nn import ShardedDataParallel, FullyShardedDataParallel


def run_iter(model, optimizer, criterion):
    img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda()
    label = torch.randint(0, NUM_CLASS, (BATCH_SIZE, )).cuda()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(img)
        loss = criterion(out, label)
    loss.backward()
    optimizer.step()


def init(model, criterion, stage):
    if stage > 0:
        optimizer = OSS(params=model.parameters(), optim=OPTIM, lr=0.001)

    if stage < 2:
        model = DDP(model, device_ids=[dist.get_rank()])
    elif stage == 2:
        model = ShardedDataParallel(model, optimizer)
    elif stage == 3:
        model = FullyShardedDataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion
