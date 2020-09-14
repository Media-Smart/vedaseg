import torch
from torch.utils.data import Sampler

from .registry import NON_DISTRIBUTED_SAMPLERS


@NON_DISTRIBUTED_SAMPLERS.register_module
class DefaultSampler(Sampler):
    """Default non-distributed sampler."""

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.dataset)).tolist())
        else:
            return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
