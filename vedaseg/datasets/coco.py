import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import json

from vedaseg.datasets.base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class CocoDataset(BaseDataset):
    def __init__(self, root, ann_file, img_prefix='', transform=None,
                 multi_label=False):
        super().__init__()
        self.multi_label = multi_label
        self.root = root
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.transform = transform
        if self.root is not None:
            self.img_prefix = os.path.join(self.root, self.img_prefix)

        self.data = json.load(
            open(os.path.join(self.root, 'annotations', self.ann_file), 'r'))

        self.load_annotations()
        logger.debug('Total of images is {}'.format(len(self.data_infos)))

    def load_annotations(self):
        self.cat_ids = [cat['id'] for cat in self.data['categories']]
        self.numclass = len(self.cat_ids)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids, self.data_infos = [], []
        self.imgToAnns = defaultdict(list)

        for img in self.data['images']:
            self.img_ids.append(img['id'])
            img['filename'] = os.path.join(self.img_prefix, img['file_name'])
            self.data_infos.append(img)

        for ann in self.data['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def get_ann_info(self, img_info):
        img_id = img_info['id']
        ann_info = [ann for ann in self.imgToAnns[img_id]]
        return self._parse_ann_info(img_info, ann_info)

    def generate_mask(self, shape, ann_info):
        h, w, c = shape
        if self.multi_label:
            masks = [np.zeros((h, w), np.uint8) for _ in range(self.numclass)]
            for m, l in zip(ann_info['masks'], ann_info['labels']):
                for m_ in m:
                    m_ = np.array(m_).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(masks[l], [m_], 1)
        else:
            mask = np.zeros((h, w), np.uint8)
            for m, l in zip(ann_info['masks'], ann_info['labels']):
                for m_ in m:
                    m_ = np.array(m_).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [m_], int(l + 1))
            masks = [mask]
        return masks

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)

        img = cv2.imread(img_info['filename']).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self.generate_mask(img.shape, ann_info)
        image, masks = self.process(img, masks)
        return image, masks.long()

    def __len__(self):
        return len(self.data_infos)
