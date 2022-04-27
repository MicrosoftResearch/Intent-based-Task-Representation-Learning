#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
import gzip
import lzma
import logging
import random

import numpy as np
import torch


def init_logger(name: Optional[str] = 'logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def open_helper(filepath: str,
                mode: Optional[str] = 'r',
                encoding: Optional[str] = 'utf_8'):
    if filepath.endswith('.gz') or filepath.endswith('.xz'):
        if mode in {'r', 'w'}:
            mode += 't'
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode=mode, encoding=encoding)
    if filepath.endswith('.xz'):
        return lzma.open(filepath, mode=mode, encoding=encoding)
    return open(filepath, mode=mode, encoding=encoding)


def batchfy(reader, batchsize: int):
    buff = []
    for instance in reader:
        buff.append(instance)
        if len(buff) == batchsize:
            yield buff
            buff = []
    if len(buff) > 0:
        yield buff


def set_seed(seed: Optional[int] = 42, cuda: Optional[int] = -1):
    random.seed(seed)
    np.random.seed(seed)
    # tensorflow.random.set_seed(seed)
    torch.manual_seed(seed)
    if cuda >= 0:
        torch.cuda.manual_seed_all(seed)
