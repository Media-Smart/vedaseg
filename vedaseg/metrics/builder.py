from ..utils import build_from_cfg
from .metrics import Compose
from .registry import METRICS


def build_metrics(cfg):
    mtcs = []
    for icfg in cfg:
        mtc = build_from_cfg(icfg, METRICS)
        mtcs.append(mtc)
    metrics = Compose(mtcs)

    return metrics
