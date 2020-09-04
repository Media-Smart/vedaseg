from ...utils import build_from_cfg
from .registry import SAMPLERS


def build_sampler(cfg, default_args=None):
    sampler = build_from_cfg(cfg, SAMPLERS, default_args)

    return sampler
