import os
import numpy as np
import yaml
import time
import argparse
import json
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

import models
import util.misc as misc
from engine_pretrain import train_one_epoch
from util.dataset import build_dataset, get_dataloader
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG pre-training args')
    parser.add_argument('--config_path', default='./configs/pretrain.yaml', type=str, help='YAML config file path for pretraining')

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

    # seeds
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # ecg dataset & dataloader
    train_dataset = build_dataset(config['dataset'], split='train')
    data_loader_train = get_dataloader(train_dataset, mode='train', **config['dataloader'])

    # define the model
    model_name = config['model_name']
    if model_name in models.__dict__:
        model = models.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')
    model.to(device)
    
    optimizer = get_optimizer_from_config(config['train'], model)
    loss_scaler = NativeScaler()

    for epoch in range(config['start_epoch'], config['train']['epochs']):
        train_one_epoch(model,
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
