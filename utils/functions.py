import torch
import numpy as np
import math
from scipy.io import loadmat
from typing import Tuple
import torch.nn as nn
from typing import Tuple, Dict
import torchmetrics

# load ecg and ecg header
def load_data(filename: str):
    x = loadmat(filename + '.mat')
    data = np.asarray(x['val'], dtype=np.float64)
    with open(filename + '.hea', 'r') as f:
        header_data = f.readlines()
    return data, header_data

def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch, end_epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (end_epoch - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

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

def get_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_name = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    kwargs = config.get('optimizer_kwargs', {})
    if opt_name == "sgd":
        momentum = kwargs.get('momentum', 0)
        return torch.optim.SGD(model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay)
    elif opt_name == "adamw":
        betas = kwargs.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.AdamW(model.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")



def build_metric_fn(config: dict) -> Tuple[torchmetrics.Metric, Dict[str, float]]:
    common_metric_fn_kwargs = {"task": config["task"],
                               "compute_on_cpu": config["compute_on_cpu"],
                               "sync_on_compute": config["sync_on_compute"]}
    if config["task"] == "multiclass":
        assert "num_classes" in config, "num_classes must be provided for multiclass task"
        common_metric_fn_kwargs["num_classes"] = config["num_classes"]
    elif config["task"] == "multilabel":
        assert "num_labels" in config, "num_labels must be provided for multilabel task"
        common_metric_fn_kwargs["num_labels"] = config["num_labels"]

    metric_list = []
    for metric_class_name in config["target_metrics"]:
        if isinstance(metric_class_name, dict):
            # e.g., {"AUROC": {"average": macro}}
            assert len(metric_class_name) == 1, f"Invalid metric name: {metric_class_name}"
            metric_class_name, metric_fn_kwargs = list(metric_class_name.items())[0]
            metric_fn_kwargs.update(common_metric_fn_kwargs)
        else:
            metric_fn_kwargs = common_metric_fn_kwargs
        assert isinstance(metric_class_name, str), f"metric name must be a string: {metric_class_name}"
        assert hasattr(torchmetrics, metric_class_name), f"Invalid metric name: {metric_class_name}"
        metric_class = getattr(torchmetrics, metric_class_name)
        metric_fn = metric_class(**metric_fn_kwargs)
        metric_list.append(metric_fn)
    metric_fn = torchmetrics.MetricCollection(metric_list)

    best_metrics = {
        k: -float("inf") if v.higher_is_better else float("inf")
        for k, v in metric_fn.items()
    }

    return metric_fn, best_metrics


def is_best_metric(metric_class: torchmetrics.Metric,
                   prev_metric: float,
                   curr_metric: float) -> bool:
    # check "higher_is_better" attribute of the metric class
    higher_is_better = metric_class.higher_is_better
    if higher_is_better:
        return curr_metric > prev_metric
    else:
        return curr_metric < prev_metric
