import torch
import logging
import os.path as osp
import numpy as np
from collections.abc import Iterable

from vedaseg.utils.checkpoint import load_checkpoint, save_checkpoint

from .registry import RUNNERS

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """
    def __init__(self,
                 loader,
                 model,
                 criterion,
                 metric,
                 optim,
                 lr_scheduler,
                 max_epochs,
                 workdir,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu

        self.train_score = None
        self.val_score = None

        self._iter = 0
        self.epoch = 0

    def __call__(self):
        #self.validate_epoch()
        #exit(0)
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_epoch()
            self.save_checkpoint(self.workdir)
            if (epoch + 1) % self.trainval_ratio == 0 \
                    and self.loader.get('val'):
                self.validate_epoch()

    def train_epoch(self):
        logger.info('Epoch %d, Start Training' % self.epoch)
        #self.save_checkpoint(self.workdir)
        for img, label in self.loader['train']:
            self.train_batch(img, label)

        self.lr_scheduler.step()

    def validate_epoch(self):
        logger.info('Epoch %d, Start Validating' % self.epoch)
        score = None
        img_num = 0
        for img, label in self.loader['val']:
            img_num += img.shape[0]
            tscore = self.validate_batch(img, label)
            if score is not None:
                score += tscore
            else:
                score = tscore
            #logging.info('img_num %d, total batch %d' % (img_num, len(self.loader['val'])))
            if img_num > 1200:
                pass
            #break
        score /= img_num
        score = score.max(0)[0].cpu().numpy()
        logging.info('Dice score is %s' % str(score))

    def train_batch(self, img, label):
        self.model.train()
        self.optim.zero_grad()

        if self.gpu:
            img = img.cuda()
            label = label.cuda()
        pred = self.model(img)
        loss = self.criterion(pred, label)

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            prob = pred.sigmoid()
            
            '''
            import matplotlib.pyplot as plt
            pred = (prob[0]).permute(1, 2, 0).float().cpu().numpy()[:, :, 0]
            im = img[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()
            label_ = label[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()[:, :, 0]
            import random
            random_num = random.randint(0, 1000)
            pred_name = 'output/%d_pred.jpg' % random_num
            plt.imsave(pred_name, pred, cmap='Greys')
            im_name = 'output/%d.jpg' % random_num
            plt.imsave(im_name, im, cmap='Greys')
            label_name = 'output/%d_gt.jpg' % random_num
            plt.imsave(label_name, label_, cmap='Greys')
            '''

            scores = self.metric(prob, label)
            #print(score / img.shape[0])
            score, _ = scores.max(0)
            score = score.cpu().numpy() / img.shape[0]
            if self.train_score is not None:
                self.train_score = 0.9 * self.train_score + 0.1 * score
            else:
                self.train_score = score
        #print('Train, %s' % score)
        self._iter += 1
        if self.iter % 1 == 0:
            logger.info(
                'Train, Epoch %d, Iter %d, LR %s, Loss %.4f, Avg Dice %s' %
                (self.epoch, self.iter, self.lr, loss.item(),
                 self.train_score))

    def validate_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()
            pred = self.model(img)

            prob = pred.sigmoid()

            score, _ = self.metric(prob, label).max(0)
            score = score.cpu().numpy() / img.shape[0]
            if self.val_score is None:
                self.val_score = score
            else:
                self.val_score = 0.1 * score + 0.9 * self.val_score

            #print('Validate, %s' % (self.val_score))

            score = self.metric(prob, label)
            #score, _ = self.metric(pred.sigmoid(), label).max(0)
            #score = score.cpu().numpy() / img.shape[0]
        return score

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if self.epoch % self.snapshot_interval == 0 or self.epoch == self.max_epochs:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            linkpath = osp.join(out_dir, 'latest.pth')
            optimizer = self.optim if save_optimizer else None
            logger.info('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

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
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               resume_lr=True,
               resume_epoch=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint,
                                              map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self._iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']
