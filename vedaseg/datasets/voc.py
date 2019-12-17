from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from PIL import Image
import os
import cv2
import logging

from .registry import DATASETS
from .base import BaseDataset

logger = logging.getLogger()


@DATASETS.register_module
class VOCDataset(BaseDataset):
    """
    """
    def __init__(self, imglist_name, root, transform):
        super().__init__()

        imglist_fp = '%s/ImageSets/Segmentation/%s' % (root, imglist_name)
        self.imglist = self.read_imglist(imglist_fp)

        logger.debug('Total of images is %d' % len(self.imglist))
        self.root = root
        self.transform = transform

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        mask_fp = os.path.join(self.root, 'EncodeSegmentationClass', imgname) + '.png'
        img = cv2.imread(img_fp).astype(np.float32)
        mask = np.array(Image.open(mask_fp), dtype=np.float32)
        #mask = np.zeros((img.shape[0], img.shape[1])) #np.array(Image.open(mask_fp))
        image, mask = self.process(img, mask)
        mask = mask.long()

        return image, mask

    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll

