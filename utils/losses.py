from typing import Tuple
import torch.nn as nn


def build_loss_fn(config: dict) -> Tuple[nn.Module, nn.Module]:
    loss_name = config['name']
    if loss_name == "cross_entropy": # for multi-class
        loss_fn = nn.CrossEntropyLoss()
        output_act = nn.Softmax(dim=-1)
    elif loss_name == "bce":         # for multi-label
        loss_fn = nn.BCEWithLogitsLoss()
        output_act = nn.Sigmoid()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
    return loss_fn, output_act
