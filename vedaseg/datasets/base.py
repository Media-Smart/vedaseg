from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """

    def __init__(self):
        self.transform = None

    def process(self, image, mask):
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
