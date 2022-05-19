#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from tqdm import tqdm
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
    COLS = ['task', 'task.tok', 'list', 'list.tok', 'task.full.tok', 'task.predicate', 'task.full.freq']
    data = df[COLS].to_dict('records')
    for i, record in enumerate(data):
        if isinstance(record['task.full.tok'], str):
            data[i]['task.full.tok'] = record['task.full.tok'].split('@@@')
        else:
            data[i]['task.full.tok'] = []
        if isinstance(record['task.predicate'], str):
            data[i]['task.predicate'] = record['task.predicate'].split('@@@')
        else:
            data[i]['task.predicate'] = []

    if args.path_comet:
        comet = {}
        for path_comet in args.path_comet:
            if verbose:
                logger.info(f'Load {path_comet}')
            with open_helper(path_comet) as f:
                for line in f:
                    taskname, rel, *generated = line.strip('\n').split('\t')
                    if taskname not in comet:
                        comet[taskname] = {}
                    comet[taskname][rel] = list(set([txt for txt in generated if txt != 'none' and len(txt) > 0]))
        for i, record in enumerate(tqdm(data)):
            for taskname in record['task.full.tok']:
                for rel, generated in comet.get(taskname, {}).items():
                    if f'comet-{rel}' not in data[i]:
                        data[i][f'comet-{rel}'] = []
                    data[i][f'comet-{rel}'].append(generated)

    with open_helper(args.path_output, 'w') as f:
        f.write('\n'.join(json.dumps(record) for record in tqdm(data)) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('DataCreation')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--comet', nargs='*', dest='path_comet', help='path to COMET output')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True,
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
