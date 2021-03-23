import torch.nn as nn

from vedaseg.utils import build_from_cfg
from .registry import UTILS


def build_module(cfg, default_args=None):
    try:
        module = build_from_cfg(cfg, UTILS, default_args)
    except KeyError as error:
        if ' is not in the ' not in error.args[0]:
            raise KeyError from error
        if ' registry' not in error.args[0]:
            raise KeyError from error
        module = build_torch_nn(cfg, default_args=default_args)

    return module


def build_torch_nn(cfg, default_args=None):
    module = build_from_cfg(cfg, nn, default_args, 'module')
    return module
