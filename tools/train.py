import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedaseg.runners import TrainRunner
from vedaseg.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--distribute', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    os.makedirs(workdir, exist_ok=True)

    train_cfg = cfg['train']
    inference_cfg = cfg['inference']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir
    common_cfg['distribute'] = args.distribute

    runner = TrainRunner(train_cfg, inference_cfg, common_cfg)
    runner()


if __name__ == '__main__':
    main()
