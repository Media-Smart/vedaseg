import argparse
import sys
sys.path.insert(0, '../vedaseg')

from vedaseg.assemble import assemble
import vedaseg.utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a semantic segmentatation model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    utils.set_random_seed(0)
    runner = assemble(cfg_fp)
    runner()


if __name__ == '__main__':
    main()
