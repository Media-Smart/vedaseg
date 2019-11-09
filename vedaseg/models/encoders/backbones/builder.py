from vedaseg.utils import build_from_cfg

from .registry import BACKBONES


def build_backbone(cfg, default_args=None):
    #import pdb
    #pdb.set_trace()
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone
