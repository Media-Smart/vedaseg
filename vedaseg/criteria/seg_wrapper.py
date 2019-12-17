import torch.nn as nn
import torch.nn.functional as F
from vedaseg.utils import build_from_cfg

from .registry import CRITERIA


class CriterionWrapper(nn.Module):
    """LossWrapper

        Args:
    """
    def __init__(self, cfg):
        super().__init__()
        #self.criterion = build_from_cfg(cfg, nn, method='module')
        self.criterion = build_from_cfg(cfg, CRITERIA, src='registry')

    def forward(self, pred, target):
        pred = F.interpolate(pred, target.shape[2:])
        return self.criterion(pred, target)
