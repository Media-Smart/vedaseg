import numpy as np
import os
import glob
import cv2
from PIL import Image


def encode_mask(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(np.int32)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.int32)
    return label_mask


def main():
    root = 'VOCdevkit/VOC2012'
    src_name = 'SegmentationClass'
    dst_name = 'EncodeSegmentationClassPart'
    src_dir = '%s/%s' % (root, src_name)
    dst_dir = '%s/%s' % (root, dst_name)
    os.makedirs(dst_dir)
    items = glob.glob('%s/*.png' % src_dir)
    total = len(items)
    for idx, item in enumerate(items):
        print('%d/%d' % (idx, total))
        new_item = item.replace(src_name, dst_name)
        new_mask = np.array(Image.open(item))
        Image.fromarray(new_mask.astype(dtype=np.uint8)).save(new_item, 'PNG')


if __name__ == '__main__':
    main()
