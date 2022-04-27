#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import argparse

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from torch import no_grad

from baseline_encoders.dataset_readers import read_CoTL
from baseline_encoders.utils import batchfy
from baseline_encoders.utils import init_logger
from baseline_encoders.utils import open_helper

verbose = False

def main(args):
    global verbose
    verbose = args.verbose

    # Load a model
    if verbose:
        logger.info(f'Model type: {args.model_type}')
    model = SentenceTransformer(args.model_type)
    if args.cuda >= 0:
        model.to(args.cuda)

    # Read a dataset
    if verbose:
        logger.info(f'Data type: {args.data_type}')
    if args.data_type == 'CoTL':
        reader = read_CoTL(args.path_input)
    else:
        raise NotImplementedError

    # Encode text
    if verbose:
        logger.info(f'Encode text')
        logger.info(f'How to concatenate task and list names: {args.concat}')
        logger.info(f'How to aggregate embeddings: {args.pooling}')
    reader_ = batchfy(reader, batchsize=args.batchsize)
    encoded = []

    with no_grad():
        for batch in tqdm(reader_):
            tasks1, lists1, tasks2, lists2, _ = zip(*batch)
            encoded_ = []
            for tasks, lists in [(tasks1, lists1), (tasks2, lists2)]:
                # Encode
                if lists[0] is None:  # No list
                    pooled_embs = model.encode(tasks)
                else:
                    if args.concat == 'input':
                        pooled_embs = model.encode([f'{l} {t}'for l, t in zip(lists, tasks)])
                    else:
                        raise NotImplementedError

                encoded_ += [tasks, lists, pooled_embs.tolist()]
            encoded += list(zip(*encoded_))

    # Output
    if verbose:
        logger.info(f'Write {len(encoded)} embeddings to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        dim = len(encoded[0][-1]) * 2
        f.write(f'{len(encoded)} {dim}\n')
        for task1, list1, emb1, task2, list2, emb2 in encoded:
            emb1 = ' '.join(map(str, emb1))
            emb2 = ' '.join(map(str, emb2))
            if list1 is None and list2 is None:
                f.write(f'{task1.replace(" ", "_")} {task2.replace(" ", "_")} {emb1} {emb2}\n')
            else:
                f.write(f'{task1.replace(" ", "_")}@@@{list2.replace(" ", "_")}@@@{task1.replace(" ", "_")}@@@{list2.replace(" ", "_")} {emb1} {emb2}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('BaselineTransformers')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--data-type', choices=['CoTL'], required=True, help='Dataset type')
    parser.add_argument('--dummy-list', help='dummy to-do list name for UIT')
    parser.add_argument('--model-type', default='all-mpnet-base.v2', help='SBERT model type')
    parser.add_argument('--concat', choices=['input', 'output'], default='input', help='how to concatenate task and list names. output will double the dimension size')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'max', 'sum'], default='mean', help='how to aggregate token embeddings')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
