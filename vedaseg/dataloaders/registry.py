from torch.utils.data import DataLoader

from ..utils import Registry

DATALOADERS = Registry('dataloader')

DATALOADERS.register_module(DataLoader)