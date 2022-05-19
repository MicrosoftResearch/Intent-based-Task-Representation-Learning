 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import numpy as np

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    assert args.num_valid > 0
    assert args.num_test > 0

    with open_helper(args.path_input) as f:
        data = [json.loads(line) for line in f]
    if verbose:
        logger.info(f'Read {len(data)} rows from {args.path_input}')

    tasknames = sorted(list({record['task.tok'] for record in data}))
    if verbose:
        logger.info(f'{len(tasknames)} unique task names')

    assert args.num_test + args.num_valid < len(tasknames)

    np.random.seed(42)
    np.random.shuffle(tasknames)
    valid_tasknames = set(tasknames[:args.num_valid])
    tasknames = tasknames[args.num_valid:]
    test_tasknames = set(tasknames[:args.num_test])
    train_tasknames = set(tasknames[args.num_test:])

    for split, tasknames in zip(['trn', 'vld', 'tst'],
                                [train_tasknames,
                                 valid_tasknames,
                                 test_tasknames]):
        filepath = args.path_output + f'{split}.json'
        records = [record for record in data if record['task.tok'] in tasknames]
        if verbose:
            logger.info(f'{split}: {len(records)} rows -> {filepath}')
        with open_helper(filepath, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('Split')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--num-valid', type=int, default=1000)
    parser.add_argument('--num-test', type=int, default=1000)
    parser.add_argument('-o', '--output', dest='path_output', required=True,
                        help='path to an output file (prefix)')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
