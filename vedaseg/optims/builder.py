import torch.optim as torch_optim

from vedaseg.utils import build_from_cfg


def build_optim(cfg, default_args=None):
    optim = build_from_cfg(cfg, torch_optim, default_args, 'module')
    return optim
