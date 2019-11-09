import torch
import logging
import os

import vedaseg.utils as utils
from vedaseg.loggers import build_logger
from vedaseg.datasets import build_dataset
from vedaseg.datasets.transforms.builder import build_transform
from vedaseg.dataloaders import build_dataloader
from vedaseg.models import build_model
from vedaseg.criteria import build_criterion
from vedaseg.optims import build_optim
from vedaseg.lr_schedulers import build_lr_scheduler
from vedaseg.runner import build_runner


def assemble(cfg_fp):
    _, fullname = os.path.split(cfg_fp)
    fname, ext = os.path.splitext(fullname)

    cfg = utils.Config.fromfile(cfg_fp)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    root_workdir = cfg.pop('root_workdir')
    cfg['workdir'] = os.path.join(root_workdir, fname)

    os.makedirs(cfg['workdir'], exist_ok=True)

    # 1. logging
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    logger.info('Assemble, Step 1, Build Dataset')
    # 2. data
    ## 2.1 dataset
    train_tf = build_transform(cfg['data']['train']['transforms'])
    train_dataset = build_dataset(cfg['data']['train']['dataset'],
                                  dict(transform=train_tf))

    if cfg['data'].get('val'):
        val_tf = build_transform(cfg['data']['val']['transforms'])
        val_dataset = build_dataset(cfg['data']['val']['dataset'],
                                    dict(transform=val_tf))

    logger.info('Assemble, Step 2, Build Dataloader')
    # 2.2 dataloader
    train_loader = build_dataloader(cfg['data']['train']['loader'],
                                    dict(dataset=train_dataset))
    loader = {'train': train_loader}
    if cfg['data'].get('val'):
        val_loader = build_dataloader(cfg['data']['val']['loader'],
                                      dict(dataset=val_dataset))
        loader['val'] = val_loader

    logger.info('Assemble, Step 3, Build Model')
    # 3. model
    model = build_model(cfg['model'])
    if cfg['gpu_id']:
        logger.info('Using GPU %s' % cfg['gpu_id'])
        model.cuda()
    else:
        logger.info('Using CPU')

    logger.info('Assemble, Step 4, Build Criterion')
    # 4. criterion
    criterion = build_criterion(cfg['criterion'])

    logger.info('Assemble, Step 5, Build Optimizer')
    # 5. optim
    optim = build_optim(cfg['optimizer'], dict(params=model.parameters()))

    logger.info('Assemble, Step 6, Build LR Scheduler')
    # 5. optim
    # 6. lr scheduler
    lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'],
                                      dict(optimizer=optim))

    logger.info('Assemble, Step 7, Build Runner')
    # 5. optim
    # 7. runner
    runner = build_runner(
        cfg['runner'],
        dict(loader=loader,
             model=model,
             criterion=criterion,
             metric=utils.dice_score,
             optim=optim,
             lr_scheduler=lr_scheduler,
             workdir=cfg['workdir'],
             gpu=cfg['gpu_id']))
    if cfg['resume']:
        runner.resume(**cfg['resume'])

    return runner
