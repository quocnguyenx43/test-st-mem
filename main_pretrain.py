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

from models.st_mem import st_mem_vit_base_dec256d4b
import utils.misc as misc
import utils.functions as f
from engine_pretrain import train_one_epoch
from utils.dataset import build_dataset, get_data_loader
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import get_optimizer_from_config


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG pre-training args')
    parser.add_argument('--config_path', default='./configs/pretrain.yaml', type=str)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v
    return config


def main(config):
    # configs
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    device = torch.device(config['device'])
    f.setup_seed(config['seed'])

    # dirs
    if config['output_dir']:
        output_dir = config['output_dir'] + config['exp_name'] + '/'
        log_dir = output_dir + 'log/'
        model_dir = output_dir + 'models/'

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        print('There not log_writer!')
        output_dir = None
        log_dir = None
        model_dir = None
        log_writer = None

    # ecg dataset & data loader
    dataset_train = build_dataset(config['dataset'], split='train')
    data_loader_train = get_data_loader(dataset_train, mode='train', **config['dataloader'])

    # model
    model = st_mem_vit_base_dec256d4b(**config['model']) 
    model.to(device)
    optimizer = get_optimizer_from_config(config['train'], model)
    loss_scaler = NativeScaler()

    for epoch in range(0, config['train']['total_epochs']):
        train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer,
            config['train']
        )



if __name__ == "__main__":
    config = parse()
    main(config)
