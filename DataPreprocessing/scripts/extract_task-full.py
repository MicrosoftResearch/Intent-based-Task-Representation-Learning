#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
import argparse

import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    df = pd.read_csv(args.path_input, delimiter='\t', low_memory=False)
    if verbose:
        logger.info(f'Read {len(df)} rows from {args.path_input}')
    texts = set(chain.from_iterable([text.split('@@@')
                                     for text in df['task.full.tok'].unique()]))

    if verbose:
        logger.info(f'Write {len(texts)} to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write('\n'.join(sorted(list(texts))) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('task.full')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
