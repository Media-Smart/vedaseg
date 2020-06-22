import logging
import os

import cv2
import numpy as np

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


def read_annot(fp):
    res = {}
    with open(fp, 'r') as fd:
        for line in fd:
            segs = line.strip().split(' ')
            img_name, type_, x, y, w, h = segs
            type_, x, y, w, h = int(type_), int(x), int(y), int(w), int(h)
            res.setdefault(img_name, [])
            res[img_name].append([type_, x, y, w, h])
    res_list = []
    for k, v in res.items():
        res_list.append((k, v))
    return res_list


@DATASETS.register_module
class CoilDataset(BaseDataset):
    """
    """

    def __init__(self, filename, data_folder, transform):
        super().__init__()

        annot_path = '%s/%s' % (data_folder, filename)
        self.annots = read_annot(annot_path)
        logger.debug('train_df sample is\n %s' % self.annots)
        self.root = data_folder
        self.transform = transform

    def __getitem__(self, idx):
        img_name, bboxes = self.annots[idx]
        img_path = os.path.join(self.root, 'images', img_name)
        orig_img = cv2.imread(img_path)
        H, W, _ = orig_img.shape
        img = np.zeros((1024, 1024, 3), dtype=np.float32)
        img[:H, :W, :] = orig_img

        mask = make_mask(bboxes, W, H)

        img, mask = self.process(img, mask)

        return img, mask

    def __len__(self):
        return len(self.annots)


def make_mask(bboxes, W, H):
    mask = np.zeros((1, H, W), dtype=np.float32)  # 4:class 1～4 (ch:0～3)

    for bbox in bboxes:
        type_, x, y, w, h = bbox
        mask[0, y: y + h, x: x + w] = 1

    return mask
