from vedaseg.utils import build_from_cfg

from .registry import HEADS


def build_head(cfg, default_args=None):
    #import pdb
    #pdb.set_trace()
    head = build_from_cfg(cfg, HEADS, default_args)
    return head
