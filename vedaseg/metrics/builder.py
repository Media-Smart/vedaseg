from ..utils import build_from_cfg
from .registry import METRICS
from .metrics import Compose


def build_metrics(cfg):
    mtcs = []
    for icfg in cfg:
        mtc = build_from_cfg(icfg, METRICS)
        mtcs.append(mtc)
    metrics = Compose(mtcs)

    return metrics
