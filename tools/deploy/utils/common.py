import os

import cv2
import torch
import numpy as np

from volksdep.metrics import Metric as BaseMetric
from volksdep.datasets import Dataset
from volksdep.calibrators import EntropyCalibrator, EntropyCalibrator2, \
    MinMaxCalibrator


CALIBRATORS = {
    'entropy': EntropyCalibrator,
    'entropy_2': EntropyCalibrator2,
    'minmax': MinMaxCalibrator,
}


class CalibDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        super().__init__()

        self.root = images_dir
        self.samples = os.listdir(images_dir)
        self.transform = transform

    def __getitem__(self, idx):
        image_file = os.path.join(self.root, self.samples[idx])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        dummy_mask = np.zeros((h, w))

        if self.transform:
            image = self.transform(image=image, masks=[dummy_mask])['image']

        return image

    def __len__(self):
        return len(self.samples)


class Metric(BaseMetric):
    def __init__(self, metric, postprocess):
        self.metric = metric
        self.postprocess = postprocess

    def __call__(self, preds, targets):
        self.metric.reset()
        preds = self.postprocess(torch.from_numpy(preds)).numpy()
        self.metric(preds, targets)
        res = self.metric.accumulate()

        return ', '.join(['{}: {}'.format(k, np.round(v, 6)) for k, v in res.items()])

    def __str__(self):
        return self.metric.__class__.__name__.lower()