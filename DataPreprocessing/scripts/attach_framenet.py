#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
import argparse
import json

from tqdm import tqdm
import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def read_framenet(filepath: str):
    table = {}
    df = pd.read_csv(filepath, delimiter='\t')
    for _, row in df.iterrows():
        root_frames = row['root'].split('@@@')
        if 'Event' not in root_frames:
            continue
        key = row['text']
        frame = row['frame']
        try:
            cores = row['cores'].split(', ')
        except AttributeError:
            cores = []
        try:
            non_cores = row['non-cores'].split(', ')
        except AttributeError:
            cores = []
        if key in table:
            table[key][0].append(frame)
            table[key][1] += cores
            table[key][2] += non_cores
        else:
            table[key] = [[frame], cores, non_cores]

    for key, vals in table.items():
        table[key] = {
            'frame': sorted(list(set(vals[0]))),
            'core': sorted(list(set(vals[1]))),
            'noncore': sorted(list(set(vals[2])))
        }
    if verbose:
        logger.info(f'Found {len(table)} texts with at least one frame')

    return table

def main(args):
    global verbose
    verbose = args.verbose

    if verbose:
        logger.info(f'Read {args.path_framenet}')
    table = read_framenet(args.path_framenet)

    if verbose:
        logger.info(f'IN: {args.path_input}')
        logger.info(f'OUT: {args.path_output}')
    of = open_helper(args.path_output, 'w')
    with open_helper(args.path_input) as f:
        for line in tqdm(f):
            record = json.loads(line)
            record['framenet'] =  [table.get(task, None) for task in record['task.full.tok']]
            of.write(json.dumps(record) + '\n')

    of.close()

    return 0


if __name__ == '__main__':
    logger = init_logger('FrameNet')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--framenet', dest='path_framenet', required=True, help='path to FrameNet identification results')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file. If not provided, write to stdout')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
