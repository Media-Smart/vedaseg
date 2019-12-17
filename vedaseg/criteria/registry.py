import torch.nn as nn

from vedaseg.utils import Registry

CRITERIA = Registry('criterion')

BCEWithLogitsLoss = nn.BCEWithLogitsLoss
CRITERIA.register_module(BCEWithLogitsLoss)


CrossEntropyLoss = nn.CrossEntropyLoss
CRITERIA.register_module(CrossEntropyLoss)
