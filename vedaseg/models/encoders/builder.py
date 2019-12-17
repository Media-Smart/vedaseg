import torch.nn as nn

from vedaseg.utils import build_from_cfg

from .backbones.registry import BACKBONES
from .enhance_modules.registry import ENHANCE_MODULES


def build_encoder(cfg, default_args=None):
    backbone = build_from_cfg(cfg['backbone'], BACKBONES, default_args)

    enhance_cfg = cfg.get('enhance')
    if enhance_cfg:
        enhance_module = build_from_cfg(enhance_cfg, ENHANCE_MODULES, default_args)
        encoder = nn.Sequential(backbone, enhance_module)
    else:
        encoder = backbone

    return encoder
