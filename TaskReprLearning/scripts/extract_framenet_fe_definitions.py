#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
import argparse
import json
import re

from nltk.corpus import framenet as fn
from tqdm import tqdm
import spacy

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None
r_space = re.compile(r'\s\s+')

def main(args):
    global verbose
    verbose = args.verbose

    frames = []
    for filepath in args.paths_input:
        with open_helper(filepath) as f:
            data = [json.loads(line) for line in tqdm(f)]
            frames += list(chain.from_iterable([
                list(chain.from_iterable([frame['frame'] if isinstance(frame, dict) else []
                                          for frame in record.get('framenet', [])]))
                for record in data]))
    frames = set(frames)
    if verbose:
        logger.info(f'{len(frames)} unique frames')

    fe2def = defaultdict(set)
    for frame in tqdm(frames):
        frame_obj = fn.frames(frame)[0]
        for fe, elm in frame_obj.FE.items():
            if len(elm.definition.strip()) == 0:
                if verbose:
                    logger.info(f'FE {fe} does not have a definition text')
                elm.definition = fe
            definition = r_space.sub(' ', elm.definition)
            fe2def[fe].add(definition)

    if verbose:
        logger.info(f'Write {len(fe2def)} FEs to {args.path_output}')
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    with open_helper(args.path_output, 'w') as f:
        for fe in sorted(fe2def.keys()):
            defs = sorted(list(fe2def[fe]))
            defs_tok = [' '.join(tok.text for tok in nlp(d)) for d in defs]
            defs = '@@@'.join(defs)
            defs_tok = '@@@'.join(defs_tok)
            f.write(f'{fe}\t{defs}\t{defs_tok}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('FE')
    parser = argparse.ArgumentParser()
    parser.add_argument('paths_input', nargs='+', help='paths to input files')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
