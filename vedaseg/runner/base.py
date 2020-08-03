import os
import random

import torch
import numpy as np
from torch.backends import cudnn


from ..loggers import build_logger
from ..dataloaders import build_dataloader
from ..datasets import build_dataset
from ..transforms import build_transform
from ..metrics import build_metrics


class Common:
    def __init__(self, cfg):
        # build logger
        logger_cfg = cfg.get('logger')
        if logger_cfg is None:
            logger_cfg = dict(
                handlers=(dict(type='StreamHandler', level='INFO'),))
        self.workdir = cfg.get('workdir')
        self.logger = self._build_logger(logger_cfg)

        # set gpu devices
        self.use_gpu = self._set_device(cfg.get('gpu_id', ''))

        # set cudnn configuration
        self._set_cudnn(
            cfg.get('cudnn_deterministic', False),
            cfg.get('cudnn_benchmark', False))

        # set seed
        self._set_seed(cfg.get('seed', None))

        # build metric
        if 'metrics' in cfg:
            self.metric = self._build_metric(cfg['metrics'])

    def _build_logger(self, cfg):
        return build_logger(cfg, dict(workdir=self.workdir))

    def _set_device(self, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if torch.cuda.is_available():
            self.logger.info('Use GPU {}'.format(gpu_id))
            use_gpu = True
        else:
            self.logger.info('Use CPU')
            use_gpu = False

        return use_gpu

    def _set_seed(self, seed):
        if seed:
            self.logger.info('Set seed {}'.format(seed))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _set_cudnn(self, deterministic, benchmark):
        self.logger.info('Set cudnn deterministic {}'.format(deterministic))
        cudnn.deterministic = deterministic

        self.logger.info('Set cudnn benchmark {}'.format(benchmark))
        cudnn.benchmark = benchmark

    def _build_metric(self, cfg):
        return build_metrics(cfg)

    def _build_transform(self, cfg):
        return build_transform(cfg)

    def _build_dataloader(self, cfg):
        transform = build_transform(cfg['transforms'])
        dataset = build_dataset(cfg['dataset'], dict(transform=transform))
        dataloader = build_dataloader(cfg['dataloader'], dict(dataset=dataset))

        return dataloader
