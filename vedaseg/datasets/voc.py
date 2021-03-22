import cv2
import logging
import numpy as np
import os

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class VOCDataset(BaseDataset):
    def __init__(self, root, imglist_name, transform, multi_label=False):
        if multi_label:
            raise ValueError('multi label training is only '
                             'supported by using COCO data form')
        super().__init__()

        imglist_fp = os.path.join(root, 'ImageSets/Segmentation', imglist_name)
        self.imglist = self.read_imglist(imglist_fp)

        logger.debug('Total of images is {}'.format(len(self.imglist)))

        self.root = root
        self.transform = transform

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        mask_fp = os.path.join(self.root, 'EncodeSegmentationClass',
                               imgname) + '.png'

        img = cv2.imread(img_fp).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)

        image, mask = self.process(img, [mask])

        return image, mask.long()

    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll
