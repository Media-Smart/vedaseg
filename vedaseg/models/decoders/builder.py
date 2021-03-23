import torch.nn as nn

from vedaseg.utils import build_from_cfg
from .registry import BRICKS, DECODERS


def build_brick(cfg, default_args=None):
    brick = build_from_cfg(cfg, BRICKS, default_args)
    return brick


def build_bricks(cfgs):
    bricks = nn.ModuleList()
    for brick_cfg in cfgs:
        bricks.append(build_brick(brick_cfg))
    return bricks


def build_decoder(cfg, default_args=None):
    decoder = build_from_cfg(cfg, DECODERS, default_args)
    return decoder
