import torch.optim.lr_scheduler as torch_lr_scheduler
from vedaseg.utils import build_from_cfg


def build_lr_scheduler(cfg, default_args=None):
    scheduler = build_from_cfg(cfg, torch_lr_scheduler, default_args, 'module')
    return scheduler
