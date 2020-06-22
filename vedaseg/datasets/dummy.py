import torch

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class DummyDataset(BaseDataset):
    """ DummyDataset
    """

    def __init__(self, total=4, transform=None):
        super().__init__()

        self.total = total
        self.a = torch.rand(320, 320, 3).numpy()
        self.b = torch.rand(320, 320, 7).numpy()
        self.transform = transform

    def __getitem__(self, idx):
        image, mask = self.process(self.a, self.b)

        return image, mask.transpose(0, 2)

    def __len__(self):
        return self.total
