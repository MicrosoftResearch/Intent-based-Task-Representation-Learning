#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import random

import pandas as pd
import numpy as np

from scripts.utils import init_logger

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.path_input, delimiter='\t', low_memory=False)
    if verbose:
        logger.info(f'Read {len(df)} rows from {args.path_input}')

    # Sampling
    subset = df.sample(args.n_samples)

    if verbose:
        logger.info(f'Write {len(subset)} rows to {args.path_output}')
    subset.to_csv(args.path_output, sep='\t', index=False)

    return 0


if __name__ == '__main__':
    logger = init_logger('Sample')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('-n', '--n-samples', type=int, required=True, help='number of samples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
