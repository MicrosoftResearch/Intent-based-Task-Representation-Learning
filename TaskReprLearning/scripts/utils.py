#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import logging
import lzma


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
