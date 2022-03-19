# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from SiamFDA.utils.lr_scheduler import build_lr_scheduler
from SiamFDA.utils.log_helper import init_log, print_speed, add_file_handler
from SiamFDA.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from SiamFDA.utils.model_load import load_pretrain, restore_from, get_restore_epoch
from SiamFDA.utils.average_meter import AverageMeter
from SiamFDA.utils.misc import describe, commit
from SiamFDA.models.model_builder import ModelBuilder
from SiamFDA.datasets.dataset import FDADataset
from SiamFDA.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(epoch_num):
    logger.info("build train dataset")
    # train_dataset
    if cfg.FDAM.FDAM:
        train_dataset = FDADataset(epoch_num)
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=False,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    if hasattr(model, 'module'):
        model = model.module
        logger.info('remove prefix module')
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        if cfg.ADJUST.ADJUST:
            for name, param in model.neck.named_parameters():
                param.requires_grad = True
        for name, param in model.head.named_parameters():
                param.requires_grad = True

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(current_epoch)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH


    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for i in range(start_epoch, cfg.TRAIN.EPOCH):
        if cfg.BACKBONE.TRAIN_EPOCH == i:
            logger.info('start training backbone.')
            optimizer, lr_scheduler = build_opt_lr(model.module, i)
            # logger.info("model\n{}".format(describe(model.module)))

        for idx, pg in enumerate(optimizer.param_groups):
            logger.info('epoch {} lr {}'.format(i + 1, pg['lr']))
            if rank == 0:
                tb_writer.add_scalar('lr/group{}'.format(idx + 1),
                                     pg['lr'], i + 1)
        for name, param in model.module.backbone.named_parameters():
                if param.requires_grad == True:
                    logger.info(name)
        for name, param in model.module.head.named_parameters():
                if param.requires_grad == True:
                    logger.info(name)
        for name, param in model.module.neck.named_parameters():
                if param.requires_grad == True:
                    logger.info(name)
        for idx, data in enumerate(train_loader):
            tb_idx = idx + start_epoch * num_per_epoch
            data_time = average_reduce(time.time() - end)
            if rank == 0:
                tb_writer.add_scalar('time/data', data_time, tb_idx)
            outputs = model(data)
            loss = outputs['total_loss']
            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                loss.backward()
                reduce_gradients(model)
                if rank == 0 and cfg.TRAIN.LOG_GRADS:
                    log_grads(model.module, tb_writer, idx + start_epoch * num_per_epoch)

                # clip gradient
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()
            batch_time = time.time() - end
            batch_info = {}
            batch_info['batch_time'] = average_reduce(batch_time)
            batch_info['data_time'] = average_reduce(data_time)
            for k, v in sorted(outputs.items()):
                batch_info[k] = average_reduce(v.data.item())

            average_meter.update(**batch_info)
            if rank == 0:
                for k, v in batch_info.items():
                    tb_writer.add_scalar(k, v, tb_idx)

                if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                    info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                        i + 1, (idx + 1) % num_per_epoch,
                        num_per_epoch, cur_lr)
                    for cc, (k, v) in enumerate(batch_info.items()):
                        if cc % 2 == 0:
                            info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                        else:
                            info += ("{:s}\n").format(
                                getattr(average_meter, k))
                    logger.info(info)
                    print_speed(idx + 1 + i * num_per_epoch,
                                average_meter.batch_time.avg,
                                cfg.TRAIN.EPOCH * num_per_epoch)
            end = time.time()

        if get_rank() == 0:
            torch.save(
                {'epoch': i + 1,
                 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()},
                cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (i + 1))


        lr_scheduler.step(i + 1)
        cur_lr = lr_scheduler.get_cur_lr()
        logger.info('epoch: {}'.format(i + 1))
        train_loader.dataset.epoch_num = i + 1


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()
    # dist_model = DistModule(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None


    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        resume_epoch = get_restore_epoch(cfg.TRAIN.RESUME)
        if resume_epoch > cfg.BACKBONE.TRAIN_EPOCH:
            optimizer, lr_scheduler = build_opt_lr(model, resume_epoch)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
        if resume_epoch == cfg.BACKBONE.TRAIN_EPOCH:
            optimizer, lr_scheduler = build_opt_lr(model, resume_epoch)

    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        logger.info("pretrained!!")
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    # build dataset loader
    train_loader = build_data_loader(cfg.TRAIN.START_EPOCH)

    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
