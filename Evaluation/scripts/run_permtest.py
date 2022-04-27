#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from os import path
import argparse
import logging

from mlxtend.evaluate import permutation_test
from sklearn.metrics import f1_score
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pandas as pd


verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def main(args):
    global verbose
    verbose = args.verbose

    # Read data
    df1 = pd.read_csv(args.path_system1)
    df2 = pd.read_csv(args.path_system2)
    assert len(df1) == len(df2)
    assert df1['trial'].max() == df2['trial'].max()
    assert df1['trial'].min() == df2['trial'].min()
    if verbose:
        logger.info(f'Read {len(df1)} data from {args.path_system1} and {args.path_system2}')
        logger.info(f'# of trials: {df1["trial"].max()+1}')

    task = path.basename(args.path_system1).split('_')[0]
    label_indexer = defaultdict(lambda: len(label_indexer))

    p_values = []
    for i in trange(int(df1['trial'].max())+1):
        # Extract and sort by the same keys
        subset1 = df1[df1['trial']==i].sort_values(list(df1.columns))
        subset2 = df2[df2['trial']==i].sort_values(list(df1.columns))  # baseline
        assert len(subset1) == len(subset2)
        if task in {'at', 'ld2018'}:
            y1 = (subset1['pred'] == subset1['gold']).values
            y2 = (subset2['pred'] == subset2['gold']).values  # baseline
            func = lambda x, y: np.mean(x) - np.mean(y)
        else:
            y1 = subset1['pred'].apply(lambda p: int(str(p).lower() == 'true')).values
            y2 = subset2['pred'].apply(lambda p: int(str(p).lower() == 'true')).values
            gold = subset1['gold'].apply(lambda l: int(str(l).lower() == 'true')).values
            func = lambda x, y: f1_score(y_true=gold, y_pred=x, average='binary') - f1_score(y_true=gold, y_pred=y, average='binary')
        p_value = permutation_test(y1, y2,
                                   method='approximate',
                                   func=func,
                                   paired=True,
                                   num_rounds=args.num_rounds,
                                   seed=args.seed)
        tqdm.write(f'{i}: {p_value}')
        p_values.append(p_value)

    print(','.join(map(str, p_values)))

    return 0


if __name__ == '__main__':
    logger = init_logger('PermTest')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_system1', help='path to system1 predictions')
    parser.add_argument('--base', dest='path_system2', required=True, help='path to system2 predictions')
    parser.add_argument('-n', '--num-rounds', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
