import torch.utils.data as torch_data

from vedaseg.utils import build_from_cfg


def build_dataloader(cfg, default_args):
    loader = build_from_cfg(cfg, torch_data, default_args, 'module')
    return loader
