#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import logging

import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    df = pd.read_csv(args.path_input, delimiter='\t', quoting=csv.QUOTE_NONE)
    if verbose:
        logger.info(f'Read {len(df)} rows from {args.path_input}')

    # Read processed texts
    processed = {}
    with open_helper(args.path_tagged) as f:
        for line in f:
            text, tokenized, lemma, upos, xpos = line.strip('\n').split('\t')
            processed[text] = {
                'tok': tokenized,
                'lem': lemma,
                'upos': upos,
                'xpos': xpos
                }

    # Attach
    for text_type in ['task', 'list']:
        for field in ['tok', 'lem', 'upos', 'xpos']:
            df[f'{text_type}.{field}'] = df[text_type].apply(
                lambda txt: processed.get(txt, {}).get(field, ''))

    # Output
    if verbose:
        logger.info(f'Write {len(df)} to {args.path_output}')
    df.to_csv(args.path_output, sep='\t', index=False)

    return 0


if __name__ == '__main__':
    logger = init_logger('Attach')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--tagged', dest='path_tagged', required=True,
                        help='tagged texts')
    parser.add_argument('-o', '--output', dest='path_output', required=True,
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
