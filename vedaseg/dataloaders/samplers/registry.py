from torch.utils.data import DistributedSampler

from ...utils import Registry

SAMPLERS = Registry('sampler')

SAMPLERS.register_module(DistributedSampler)
