import glob
import os
import scipy.io as io
from PIL import Image
import numpy as np


def main():
    root = 'benchmark_RELEASE/dataset'
    src_name = 'cls'
    dst_name = 'encode_cls'
    src_dir = '%s/%s' % (root, src_name)
    dst_dir = '%s/%s' % (root, dst_name)
    os.makedirs(dst_dir)
    items = glob.glob('%s/*.mat' % src_dir)
    total = len(items)
    for idx, item in enumerate(items):
        print('%d/%d' % (idx, total))
        data = io.loadmat(item)
        mask = data['GTcls'][0]['Segmentation'][0].astype(np.int32)
        new_item = item.replace(src_name, dst_name).replace('.mat', '.png')
        Image.fromarray(mask.astype(dtype=np.uint8)).save(new_item, 'PNG')


if __name__ == '__main__':
    main()
