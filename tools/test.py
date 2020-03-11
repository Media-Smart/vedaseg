import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedaseg'))

from vedaseg.assembler import assembler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a semantic segmentatation model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='test checkpoint')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint

    runner = assembler(cfg_fp, checkpoint, True)
    runner()


if __name__ == '__main__':
    main()
