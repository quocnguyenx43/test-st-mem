# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import models
import util.misc as misc
from engine_pretrain import train_one_epoch
from util.dataset import build_dataset, get_dataloader
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG self-supervised pre-training')

    parser.add_argument('--config_path',
                        default='./configs/pretrain/st_mem.yaml',
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--output_dir',
                        default="",
                        type=str,
                        metavar='DIR',
                        help='path where to save')
    parser.add_argument('--exp_name',
                        default="",
                        type=str,
                        help='experiment name')
    parser.add_argument('--resume',
                        default="",
                        type=str,
                        metavar='PATH',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def main(config):
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    # fix the seed for reproducibility
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ECG dataset
    dataset_train = build_dataset(config['dataset'], split='train')
    data_loader_train = get_dataloader(dataset_train,
                                       is_distributed=config['ddp']['distributed'],
                                       mode='train',
                                       **config['dataloader'])

    if config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None

    # define the model
    model_name = config['model_name']
    if model_name in models.__dict__:
        model = models.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')
    model.to(device)

    optimizer = get_optimizer_from_config(config['train'], model)
    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {config['train']['epochs']} epochs")
    for epoch in range(config['start_epoch'], config['train']['epochs']):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      log_writer,
                                      config['train'])


if __name__ == "__main__":
    config = parse()
    main(config)
