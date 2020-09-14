from torch.utils.data import DistributedSampler

from ...utils import get_dist_info
from .registry import DISTRIBUTED_SAMPLERS


@DISTRIBUTED_SAMPLERS.register_module
class DefaultSampler(DistributedSampler):
    """Default distributed sampler."""

    def __init__(self, dataset, shuffle=True):
        rank, num_replicas = get_dist_info()
        super().__init__(dataset, num_replicas, rank, shuffle)
