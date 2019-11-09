import albumentations as albu

from vedaseg.utils import build_from_cfg


def build_transform(cfg):
    tfs = []
    for icfg in cfg:
        tf = build_from_cfg(icfg, albu, method='module')
        tfs.append(tf)
    aug = albu.Compose(tfs)
    return aug
