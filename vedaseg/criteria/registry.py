import torch.nn as nn

from vedaseg.utils import Registry

CRITERIA = Registry('criterion')

CrossEntropyLoss = nn.CrossEntropyLoss
CRITERIA.register_module(CrossEntropyLoss)
