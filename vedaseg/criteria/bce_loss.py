import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index=-1, *args, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()

        self.ignore_index = ignore_index
        self.loss = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, pred, target):
        
        valid_mask = target != self.ignore_index
        losses = self.loss(pred[valid_mask], target[valid_mask].float())

        return losses
