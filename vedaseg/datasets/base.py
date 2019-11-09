import torch
import numpy as np
import albumentations as albu
import albumentations.pytorch as albu_pytorch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """
    def __init__(self):
        self.transform = None
        self.normalize = albu.Normalize()
        self.to_tensor = albu.pytorch.ToTensorV2()

    def process(self, image, mask):
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image_norm = self.normalize(image=image)['image']
        tensors = self.to_tensor(image=image_norm, mask=mask)
        return tensors['image'], tensors['mask']
