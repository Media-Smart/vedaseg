from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import logging

from .registry import DATASETS
from .base import BaseDataset

logger = logging.getLogger()


@DATASETS.register_module
class SteelDataset(BaseDataset):
    """
    """
    def __init__(self, filename, data_folder, transform, phase):
        super().__init__()

        df_path = '%s/%s' % (data_folder, filename)
        df = pd.read_csv(df_path)
        df['ImageId'], df['ClassId'] = df['ImageId_ClassId'].str.slice(
            0, -2), df['ImageId_ClassId'].str.slice(-1)
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',
                      columns='ClassId',
                      values='EncodedPixels')
        df['defects'] = df.count(axis=1)

        train_df, val_df = train_test_split(df,
                                            test_size=0.1,
                                            stratify=df['defects'],
                                            random_state=0)
        logger.debug('train_df sample is\n %s' % train_df.head())
        if phase == 'train':
            self.df = train_df
        else:
            self.df = val_df
        self.root = data_folder
        self.fnames = self.df.index.tolist()
        self.transform = transform

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, 'train_images', image_id)
        image = cv2.imread(image_path)

        image, mask = self.process(image, mask)
        mask = mask.permute(2, 0, 1)  # 4x256x1600

        return image, mask

    def __len__(self):
        return len(self.fnames)


def make_mask(row_id, df):
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(' ')
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                pos -= 1
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks
