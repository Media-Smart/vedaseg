import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vedaseg.runner import InferenceRunner
from vedaseg.utils import Config

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def inverse_resize(pred, image_shape):
    h, w, _ = image_shape
    reisze_h, resized_w = pred.shape[0], pred.shape[1]
    scale_factor = max(h / reisze_h, w / resized_w)
    pred = cv2.resize(pred, (
        int(reisze_h * scale_factor), int(reisze_h * scale_factor)),
                      interpolation=cv2.INTER_NEAREST)
    return pred


def inverse_pad(pred, image_shape):
    h, w, _ = image_shape
    return pred[:h, :w]


def plot_result(img, mask, cover):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Vedaseg Demo", y=0.95, fontsize=16)

    ax[0].set_title('image')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[1].set_title(f'mask')
    ax[1].imshow(mask)

    ax[2].set_title(f'cover')
    ax[2].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.show()


def result(fname,
           pred_mask,
           classes,
           multi_label=False,
           palette=None,
           show=False,
           out=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(classes), 3))
    else:
        palette = np.array(palette)
    img_ori = cv2.imread(fname)
    mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        if multi_label:
            mask[pred_mask[:, :, label] == 1] = color
        else:
            mask[pred_mask == label, :] = color

    cover = img_ori * 0.5 + mask * 0.5
    cover = cover.astype(np.uint8)

    if out is not None:
        _, fullname = os.path.split(fname)
        fname, _ = os.path.splitext(fullname)
        save_dir = os.path.join(out, fname)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'img.png'), img_ori)
        cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask)
        cv2.imwrite(os.path.join(save_dir, 'cover.png'), cover)
        if multi_label:
            for i in range(pred_mask.shape[-1]):
                cv2.imwrite(os.path.join(save_dir, classes[i] + '.png'),
                            pred_mask[:, :, i] * 255)

    if show:
        plot_result(img_ori, mask, cover)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentatation model')
    parser.add_argument('config', type=str,
                        help='config file path')
    parser.add_argument('checkpoint',
                        type=str, help='checkpoint file path')
    parser.add_argument('image',
                        type=str,
                        help='input image path')
    parser.add_argument('--show', action='store_true',
                        help='show result')
    parser.add_argument('--need_resize', action='store_true',
                        help='set true if there is LongestMaxSize in transform')
    parser.add_argument('--out', default='./result',
                        help='folder to store result images')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    multi_label = cfg.get('multi_label', False)
    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    dummy_mask = np.zeros((h, w))
    output = runner(image, [dummy_mask])
    if multi_label:
        output = output.transpose((1, 2, 0))

    if args.need_resize:
        output = inverse_resize(output, image.shape)
    output = inverse_pad(output, image.shape)

    result(args.image, output, multi_label=multi_label,
           classes=CLASSES, palette=PALETTE, show=args.show,
           out=args.out)


if __name__ == '__main__':
    main()
