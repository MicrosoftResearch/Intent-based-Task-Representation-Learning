#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import argparse
import json

from nltk.corpus import framenet as fn
import numpy as np

from scripts.utils import init_logger

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    fe2frames = defaultdict(set)
    event_frame = fn.frame('Event')
    for fe in event_frame.FE.keys():
        fe2frames[fe].add(event_frame.name)

    checked = {event_frame.name}
    buff = [rel.subFrame for rel in event_frame.frameRelations if rel.type.name == 'Inheritance']
    while len(buff) > 0:
        frame = buff.pop(0)
        if frame.name in checked:
            continue
        for fe in frame.FE.keys():
            fe2frames[fe].add(frame.name)
        checked.add(frame.name)
        for rel in frame.frameRelations:
            if rel.type.name != 'Inheritance':
                continue
            if rel.subFrame.name in checked:
                continue
            buff.append(rel.subFrame)
    N = len(checked)
    if verbose:
        logger.info(f'Checked {N} frames')
        logger.info(f'Found {len(fe2frames)} FEs')

    weights = {fe: np.log((1+N)/(1+len(frames))) + 1
               for fe, frames in fe2frames.items()}
    max_weight = max(weights.values())
    with open(args.path_output, 'w') as f:
        for fe, frames in sorted(fe2frames.items(), key=lambda t: t[0]):
            dat = {
                'label': fe,
                'weight': weights[fe],
                'normalized_weight': weights[fe] / max_weight,
                'frames': sorted(frames)
            }
            f.write(json.dumps(dat) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('Weight')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, dest='path_output', help='path to an output file (.json)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
