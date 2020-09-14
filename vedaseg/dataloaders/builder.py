from .registry import DATALOADERS
from ..utils import build_from_cfg


def build_dataloader(distributed, num_gpus, cfg, default_args=None):
    cfg_ = cfg.copy()

    samples_per_gpu = cfg_.pop('samples_per_gpu')
    workers_per_gpu = cfg_.pop('workers_per_gpu')

    if distributed:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    cfg_.update({'batch_size': batch_size,
                 'num_workers': num_workers})

    dataloader = build_from_cfg(cfg_, DATALOADERS, default_args)

    return dataloader
