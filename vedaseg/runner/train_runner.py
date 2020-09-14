import os
from collections import OrderedDict
from collections.abc import Iterable

import torch
import numpy as np

from ..optims import build_optimizer
from ..criteria import build_criterion
from ..lr_schedulers import build_lr_scheduler
from ..utils import save_checkpoint, gather_tensor, reduce_tensor
from .inference_runner import InferenceRunner


class TrainRunner(InferenceRunner):
    def __init__(self, train_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.train_dataloader = self._build_dataloader(
            train_cfg['data']['train'])

        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
            self.val_exclude_num = self.world_size - len(
                self.val_dataloader.dataset) % self.world_size
        else:
            self.val_dataloader = None

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.criterion = self._build_criterion(train_cfg['criterion'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_epochs = train_cfg['max_epochs']
        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.save_best = train_cfg.get('save_best', True)
        self.iter_based = hasattr(self.lr_scheduler, '_iter_based')

        assert self.workdir is not None
        assert self.log_interval > 0

        self.best = OrderedDict()
        self.iter = 0

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, dict(params=self.model.parameters()))

    def _build_criterion(self, cfg):
        return build_criterion(cfg)

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, dict(optimizer=self.optimizer,
                                            niter_per_epoch=len(
                                                self.train_dataloader)))

    def _train(self):
        self.metric.reset()
        self.model.train()

        self.logger.info('Epoch {}, start training'.format(self.epoch + 1))
        for idx, (image, mask) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if self.use_gpu:
                image = image.cuda()
                mask = mask.cuda()

            output = self.model(image)
            loss = self.criterion(output, mask)

            loss.backward()
            self.optimizer.step()

            self.iter += 1

            with torch.no_grad():
                output = self.compute(output)

                output = gather_tensor(output)
                mask = gather_tensor(mask)
                reduced_loss = reduce_tensor(loss.item())

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.accumulate()

            if self.iter % self.log_interval == 0:
                self.logger.info(
                    'Train, Epoch {}, Iter {}, LR {}, Loss {:.4f}, {}'.format(
                        self.epoch + 1, self.iter,
                        ['{:.4f}'.format(lr) for lr in self.lr],
                        reduced_loss, ', '.join(
                            ['{}: {}'.format(k, np.round(v, 4)) for k, v in
                             res.items()])))

            if self.iter_based:
                self.lr_scheduler.step()

        if not self.iter_based:
            self.lr_scheduler.step()

    def _val(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start validating')
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.val_dataloader):
                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                output = self.model(image)
                output = self.compute(output)

                output = gather_tensor(output)
                mask = gather_tensor(mask)

                if idx + 1 == len(
                        self.val_dataloader) and self.val_exclude_num > 0:
                    output = output[:-self.val_exclude_num]
                    mask = mask[:-self.val_exclude_num]

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.accumulate()

                if (idx + 1) % self.log_interval == 0:
                    self.logger.info('Validation, Iter {}, {}'.format(
                        idx + 1,
                        ', '.join(
                            ['{}: {}'.format(k, np.round(v, 4)) for k, v in
                             res.items()])))

        return res

    def __call__(self):
        for _ in range(self.epoch, self.max_epochs):
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            self._train()

            if self.trainval_ratio > 0 and \
                    self.epoch % self.trainval_ratio == 0 and \
                    self.val_dataloader:
                res = self._val()
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        if k not in self.best:
                            self.best[k] = 0.0
                        if self.best[k] <= res[k]:
                            self.best[k] = res[k]
                            if self.save_best and self.rank == 0:
                                self.save_checkpoint(
                                    self.workdir, 'best_{}.pth'.format(k),
                                    meta=dict(best=self.best))
                self.logger.info(', '.join(
                    ['Best {}: {}'.format(k, v) for k, v in self.best.items()]))

            if self.snapshot_interval > 0 and \
                    self.epoch % self.snapshot_interval == 0 and self.rank == 0:
                self.logger.info('Snapshot')
                self.save_checkpoint(
                    self.workdir, 'epoch_{}.pth'.format(self.epoch),
                    meta=dict(best=self.best))

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optimizer.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optimizer.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def save_checkpoint(self, dir_, filename, save_optimizer=True,
                        save_lr_scheduler=True, meta=None):
        optimizer = self.optimizer if save_optimizer else None
        lr_scheduler = self.lr_scheduler if save_lr_scheduler else None

        filepath = os.path.join(dir_, filename)
        self.logger.info('Save checkpoint {}'.format(filename))
        if meta is None:
            meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
        else:
            meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)
        save_checkpoint(self.model, filepath, optimizer, lr_scheduler, meta)

    def resume(self, checkpoint, resume_optimizer=False,
               resume_lr_scheduler=False, resume_meta=False,
               map_location='default'):
        checkpoint = self.load_checkpoint(checkpoint,
                                          map_location=map_location)

        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_meta and 'meta' in checkpoint:
            self.logger.info('Resume meta data')
            self.best = checkpoint['meta']['best']
            self.epoch = checkpoint['meta']['epoch']
            self.iter = checkpoint['meta']['iter']
            self.lr = checkpoint['meta']['lr']
