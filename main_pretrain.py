import numpy as np
import os
import yaml
import time
import argparse
import json

import torch
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

import utils.misc as m
import utils.functions as f
import utils.dataset as d
from utils.engine import train_one_epoch_pretrain

from models.st_mem import st_mem_vit_base_dec256d4b


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
    dataset_train = d.build_dataset(config['dataset'], split='train')
    data_loader_train = d.get_data_loader(dataset_train, mode='train', **config['dataloader'])

    # model
    model = st_mem_vit_base_dec256d4b(**config['model'])
    optimizer = f.get_optimizer(config['train'], model)
    loss_scaler = m.NativeScalerWithGradNormCount()
    model.to(device)

    # load previous checkpoint
    ###

    for epoch in range(0, config['train']['total_epochs']):
        train_one_epoch_pretrain(
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
