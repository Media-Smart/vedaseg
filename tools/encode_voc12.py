import glob
import os

import numpy as np
from PIL import Image


def main():
    root = 'workpiece/VOC2012'
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
