from ...utils import build_from_cfg
from .registry import NON_DISTRIBUTED_SAMPLERS, DISTRIBUTED_SAMPLERS


def build_sampler(distributed, cfg, default_args=None):
    if distributed:
        sampler = build_from_cfg(cfg, DISTRIBUTED_SAMPLERS, default_args)
    else:
        sampler = build_from_cfg(cfg, NON_DISTRIBUTED_SAMPLERS, default_args)

    return sampler
