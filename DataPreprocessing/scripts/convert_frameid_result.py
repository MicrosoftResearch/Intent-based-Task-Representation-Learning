#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import argparse

from nltk.corpus import framenet as fn
from tqdm import tqdm

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None

frame_root_cache = {}
def get_root_frame(frame):
    global frame_root_cache
    if frame.name in frame_root_cache:
        return frame_root_cache[frame.name]
    if len(frame.frameRelations) == 0:
        return [frame.name]
    result = []
    for rel in frame.frameRelations:
        if rel.type.name != 'Inheritance':
            continue
        if rel.superFrame.name == frame.name:
            continue
        result += get_root_frame(rel.superFrame)
    if len(result) == 0:
        result = [frame.name]
    frame_root_cache[frame.name] = result
    return result

frame_cache = {}
def read_sentence(tokens):
    global frame_cache
    text = ' '.join(tok[1] for tok in tokens)
    for i, tok in enumerate(tokens):
        if tok[13] != '_':
            break
    trigger = tokens[i][12]
    frame = tokens[i][13]

    # Get FEs
    cores, noncores = [], []
    try:
        frame_obj, root_frame = frame_cache[frame]
    except KeyError:
        frame_obj = fn.frames(frame)[0]
        root_frame = get_root_frame(frame_obj)
        frame_cache[frame] = (frame_obj, root_frame)
    for label, fe_obj in frame_obj.FE.items():
        if fe_obj['coreType'] == 'Core':
            cores.append(label)
        else:
            noncores.append(label)
    return text, trigger, i, frame, cores, noncores, root_frame


def main(args):
    global verbose
    verbose = args.verbose

    if verbose:
        logger.info(f'Read {args.path_input}')
    text_indexer = defaultdict(lambda: len(text_indexer))
    text2frames = defaultdict(list)
    fe_counter = defaultdict(int)
    with open_helper(args.path_input) as f:
        buff = []
        for line in tqdm(f):
            if len(line.strip('\n')) == 0:
                ret = read_sentence(buff)
                text = ret[0]
                text_indexer[text]
                text2frames[text].append(ret[1:])
                for fe in ret[-2] + ret[-1]:
                    fe_counter[fe] += 1
                buff = []
                continue
            row = line.strip('\n').split('\t')
            buff.append(row)

    if verbose:
        logger.info(f'Read {len(text_indexer)} texts')
        logger.info(f'Write to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write('\t'.join(['text', 'trigger', 'trigger_index', 'frame', 'cores', 'non-cores', 'root']) + '\n')
        for text, i in sorted(text_indexer.items(), key=lambda t: t[1]):
            for frame in text2frames[text]:
                trigger, trigger_idx = frame[0], str(frame[1])
                frame_name = frame[2]
                cores, noncores = frame[3], frame[4]
                root_frame = '@@@'.join(sorted(frame[5]))
                buff = [text, trigger, trigger_idx, frame_name]
                cores = ', '.join(f'{fe}@@@{fe_counter[fe]}' for fe in cores)
                noncores = ', '.join(f'{fe}@@@{fe_counter[fe]}' for fe in noncores)
                buff += [cores, noncores, root_frame]
                f.write('\t'.join(buff) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('ConvFrameID')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('-o', '--output', dest='path_output', help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
