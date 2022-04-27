#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any
from typing import Dict
import gzip
import json
import lzma
import logging
import random

import _jsonnet
import numpy as np
import torch


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def open_helper(filepath, mode='r', encoding='utf_8'):
    if filepath.endswith('.gz') or filepath.endswith('.xz'):
        if mode in {'r', 'w'}:
            mode += 't'
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode=mode, encoding=encoding)
    if filepath.endswith('.xz'):
        return lzma.open(filepath, mode=mode, encoding=encoding)
    return open(filepath, mode=mode, encoding=encoding)


def read_config(file_path: str, ext_vars: Dict = {}):
    with open(file_path) as f:
        jsonnet_str = f.read()
        json_str = _jsonnet.evaluate_snippet('snippet', jsonnet_str,
                                             ext_vars=ext_vars)
    return json.loads(json_str)


def dumps_config(config: Dict[str, Any]) -> str:
    items = []

    # Model type
    model_config = config.get('model', {})
    items.append(f'model={model_config.get("type")}')
    # items.append(f'intent_emb_act={model_config.get("intent_emb_act", "none")}')
    primary_output = model_config.get('primary_output', 'discrete')
    items.append(f'out={primary_output}')
    if primary_output == 'discrete':
        loss_smoothing_factor = model_config.get('primary_loss_smoothing_factor', 0)
        items.append(f'smoothing={loss_smoothing_factor}')
    else:
        loss_type = model_config['primary_loss_type']
        items.append(f'loss={loss_type}')

    # Training
    trainer_config = config.get('trainer', {})
    batch_size = trainer_config.get("batch_size", 1) * trainer_config.get('num_gradient_accumulation_steps', 1)
    items.append(f'bs={batch_size}')

    # Pretraining optimizer
    items.append(f'pt_epoch={trainer_config.get("pretraining_num_epochs", 0)}')
    optimizer_config = trainer_config.get('pretraining_optimizer', {})
    items.append(f'pt_optim={optimizer_config.get("type")}')
    items.append(f'pt_lr={optimizer_config.get("lr")}')
    items.append(f'pt_wd={optimizer_config.get("weight_decay", 0)}')
    items.append(f'pt_gn={optimizer_config.get("max_grad_norm")}')

    # Optimzier
    items.append(f'epoch={trainer_config.get("num_epochs", 1)}')
    optimizer_config = trainer_config.get('optimizer', {})
    items.append(f'optim={optimizer_config.get("type")}')
    items.append(f'lr={optimizer_config.get("lr")}')
    items.append(f'wd={optimizer_config.get("weight_decay", 0)}')
    items.append(f'gn={optimizer_config.get("max_grad_norm")}')

    # Scheduler
    scheduler_config = trainer_config.get('scheduler', {})
    items.append(f'sch={scheduler_config.get("type")}')
    items.append(f'warmup={scheduler_config.get("warmup_steps_ratio")}')

    return '_'.join(items)


def set_seed(seed=42, cuda=-1):
    random.seed(seed)
    np.random.seed(seed)
    # tensorflow.random.set_seed(seed)
    torch.manual_seed(seed)
    if cuda >= 0:
        torch.cuda.manual_seed_all(seed)

def freeze_module(module):
    """Freeze torch.nn.Module (do not calculate gradients)"""
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    """Unfreeze torch.nn.Module (calculate gradients)"""
    for p in module.parameters():
        p.requires_grad = True
