#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from os import path
from typing import Optional
import argparse
import json
import logging
import pickle
import random

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def read_UIT(filepath: str,
             listname: Optional[str] = 'inbox'):
    with open_helper(filepath) as f:
        for line in f:
            task, label = line.strip('\n').split('\t')
            yield task, listname, label


def read_AT(filepath: str):
    with open_helper(filepath) as f:
        next(f)  # Skip header
        for line in f:
            listname, task, label = line.strip('\n').split('\t')
            yield task, listname, label


def read_CoTL(filepath: str):
    with open_helper(filepath) as f:
        next(f)  # Skip header
        for line in f:
            label, task1, list1, task2, list2 = line.strip('\n').split('\t')
            list1 = list1.replace('default list', 'inbox')
            list2 = list2.replace('default list', 'inbox')
            yield task1, list1, task2, list2, label


def read_LD2018(filepath: str,
                listname: Optional[str] = 'inbox'):
    with open_helper(filepath) as f:
        for line in f:
            row = json.loads(line)
            label = row['class-label']
            task = row['instance']['panon']['text']
            yield task, listname, label


def main(args):
    global verbose
    verbose = args.verbose

    # Read data
    if verbose:
        logger.info(f'IN: {args.paths_input}')
        logger.info(f'dummy list: {args.dummy_list}')
    DATASET_READERS = {
        'at': read_AT,
        'coloc': read_CoTL,
        'cotim': read_CoTL,
        'ld2018': read_LD2018,
        'uit': read_UIT,
    }
    dataset_reader = DATASET_READERS[args.task]
    data = []
    for path_input in args.paths_input:
        if args.task in {'ld2018', 'uit'}:
            data += list(dataset_reader(path_input, listname=args.dummy_list))
            continue
        data += list(dataset_reader(path_input))
    if verbose:
        logger.info(f'{len(data)} instances')
    data = np.array(data)

    random.seed(42)
    np.random.seed(42)

    indices = np.arange(len(data))
    num_test = int(args.test_ratio * len(data))
    if verbose:
        logger.info(f'# of test instances: {num_test}')
    splits = []
    if args.stratified:
        shuf = StratifiedShuffleSplit(n_splits=args.num_trials, test_size=num_test,
                                      random_state=42)
        indexer = defaultdict(lambda: len(indexer))
        labels = np.array([indexer[record[-1].lower()] for record in data])
        indices = list(shuf.split(data, labels))
        shuf_dev = StratifiedShuffleSplit(n_splits=1, test_size=num_test,
                                          random_state=42)
        for indices_train, indices_test in indices:
            X, y = data[indices_train], labels[indices_train]
            _indices_train, _indices_dev = list(shuf_dev.split(X, y))[0]
            indices_dev = indices_train[_indices_dev]
            indices_train = indices_train[_indices_train]
            if verbose:
                logger.info(f'{len(indices_train)} {len(indices_dev)} {len(indices_test)}')
            splits.append((indices_train.tolist(), indices_dev.tolist(), indices_test.tolist()))
    else:
        for t in range(args.num_trials):
            np.random.shuffle(indices)
            indices_train, indices_test = indices[:-num_test], indices[-num_test:]
            indices_dev = indices_train[-num_test:]
            indices_train = indices_train[:-num_test]
            splits.append((indices_train.tolist(), indices_dev.tolist(), indices_test.tolist()))

    filename = f'{args.task}_r{args.test_ratio}_t{args.num_trials}.pkl'
    if args.task in {'uit', 'ld2018'}:
        filename = f'{args.task}_{args.dummy_list}_r{args.test_ratio}_t{args.num_trials}.pkl'
    path_output = path.join(args.dir_output, filename)
    if verbose:
        logger.info(f'Write {len(splits)} splits to {path_output}')
    with open(path_output, 'wb') as f:
        pickle.dump((data, splits), f)

    return 0


if __name__ == '__main__':
    logger = init_logger('GenSplits')
    parser = argparse.ArgumentParser()
    parser.add_argument('paths_input', nargs='+', help='input file paths')
    parser.add_argument('--task', choices=['uit', 'at', 'coloc', 'cotim', 'ld2018'], default='uit')
    parser.add_argument('--test', dest='test_ratio', type=float, default=0.2, help='ratio of test data')
    parser.add_argument('--stratified', action='store_true', help='use stratified random sampling')
    parser.add_argument('--dummy-list', default='inbox', help='dummy list names')
    parser.add_argument('-t', '--trials', dest='num_trials', type=int, default=20, help='number of trials')
    parser.add_argument('-o', '--output', dest='dir_output', required=True, help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
