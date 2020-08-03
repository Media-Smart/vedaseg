from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """
    CLASSES = None

    PALETTE = None

    def __init__(self, transform=None):
        self.transform = transform

    def process(self, image, masks):
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            return augmented['image'], augmented['masks']
        else:
            return image, masks
