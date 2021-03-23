from torch.optim import lr_scheduler

from vedaseg.utils import build_from_cfg
from .registry import LR_SCHEDULERS


def build_lr_scheduler(cfg, default_args=None):
    if LR_SCHEDULERS.get(cfg['type']):
        scheduler = build_from_cfg(cfg, LR_SCHEDULERS, default_args, 'registry')
    else:
        default_args = dict(optimizer=default_args.get('optimizer'))
        scheduler = build_from_cfg(cfg, lr_scheduler, default_args, 'module')

    return scheduler

