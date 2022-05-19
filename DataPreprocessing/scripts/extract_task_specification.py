#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    if verbose:
        logger.info(f'Read {args.path_pas}')
    with open_helper(args.path_pas) as f:
        data = [json.loads(line) for line in f]
    if verbose:
        logger.info(f'{len(data)} records')
    buff = {'task.tok': [], 'key': [], 'rel': []}
    for record in data:
        if not isinstance(record.get('predicate'), str) \
          or len(record['predicate']) == 0:
            continue
        arguments = record.get('arguments', {})
        for obj in arguments.get('dobj', []):
            buff['task.tok'].append(record['text'])
            buff['key'].append(obj)
            buff['rel'].append('dobj')
        for pred in arguments.get('prep', []):
            buff['task.tok'].append(record['text'])
            buff['key'].append(pred)
            buff['rel'].append('prep')
        buff['task.tok'].append(record['text'])
        buff['key'].append(record['predicate'])
        buff['rel'].append('root')
    table = pd.DataFrame.from_dict(buff)

    if verbose:
        logger.info(f'Read {args.path_input}')
    df = pd.read_csv(args.path_input, delimiter='\t')
    if verbose:
        logger.info(f'{len(df)} rows')

    df_merged = pd.merge(df, table, how='left', on='task.tok').drop_duplicates()
    if verbose:
        logger.info(f'{len(df_merged)} rows')
    df_merged = df_merged[~pd.isnull(df_merged['key'])]
    if verbose:
        logger.info(f'Dropped underspecified -> {len(df_merged)}')
        logger.info(f'Write to {args.path_output}')
    df_merged.to_csv(args.path_output, sep='\t', index=False)

    return 0


if __name__ == '__main__':
    logger = init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--pas', dest='path_pas', help='path to PAS file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
