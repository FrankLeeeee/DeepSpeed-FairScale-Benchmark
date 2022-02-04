import deepspeed
from ..configs.common import *


def run_iter(model_engine, optimizer, criterion):
    assert optimizer is None
    img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda().half()
    label = torch.randint(0, NUM_CLASS, (BATCH_SIZE, )).cuda()

    model_engine.zero_grad()
    out = model_engine(img)
    loss = criterion(out, label)
    model_engine.backward(loss)
    model_engine.step()


def init(model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters)
    return model_engine
