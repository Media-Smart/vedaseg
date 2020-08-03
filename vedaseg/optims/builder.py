import torch.optim as optims

from ..utils import build_from_cfg


def build_optimizer(cfg_optimizer, default_args=None):
    optimizer = build_from_cfg(cfg_optimizer, optims, default_args, 'module')
    return optimizer
