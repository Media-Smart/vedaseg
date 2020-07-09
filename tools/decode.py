# https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
# https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py
# https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py

import glob
import os

import numpy as np
from PIL import Image


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def main():
    root = 'VOCdevkit/VOC2012'
    src_name = 'EncodeSegmentationClass'
    dst_name = 'DecodeSegmentationClass'
    src_dir = '%s/%s' % (root, src_name)
    dst_dir = '%s/%s' % (root, dst_name)
    os.makedirs(dst_dir)
    items = glob.glob('%s/*.png' % src_dir)
    total = len(items)
    for idx, item in enumerate(items):
        print('%d/%d' % (idx, total))
        new_item = item.replace(src_name, dst_name)
        target = np.array(Image.open(item))[:, :, np.newaxis]
        cmap = color_map()[:, np.newaxis, :]
        new_im = np.dot(target == 0, cmap[0])
        for i in range(1, cmap.shape[0]):
            new_im += np.dot(target == i, cmap[i])
        new_im = Image.fromarray(new_im.astype(np.uint8))
        new_im.save(new_item)


if __name__ == '__main__':
    main()
