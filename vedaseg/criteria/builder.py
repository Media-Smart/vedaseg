from ..utils import build_from_cfg
from .registry import CRITERIA


def build_criterion(cfg):
    criterion = build_from_cfg(cfg, CRITERIA, mode='registry')
    return criterion
