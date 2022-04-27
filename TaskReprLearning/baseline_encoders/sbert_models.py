#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import argparse

from sentence_transformers import SentenceTransformer
from torch import no_grad

from baseline_encoders.utils import init_logger
from baseline_encoders.dataset_readers import read_Tasks
from baseline_encoders.dataset_readers import read_UIT
from baseline_encoders.utils import batchfy
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
    if args.data_type == 'LD2018':
        if verbose:
            logger.info(f'Dummy list: {args.dummy_list}')
        reader = read_LD2018(args.path_input, listname=args.dummy_list)
    elif args.data_type == 'tasks':
        reader = read_Tasks(args.path_input)
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
        for i, batch in enumerate(reader_):
            tasks, lists, labels = zip(*batch)

            # Encode
            if lists[0] is None:  # No list
                pooled_embs = model.encode(tasks)
            else:
                if args.concat == 'input':
                    pooled_embs = model.encode([f'{l} {t}'for l, t in zip(lists, tasks)])
                else:
                    raise NotImplementedError

            if args.data_type == 'LD2018':
                tasks = [task.replace(' ', '_') + '@@@' + label.replace(' ', '_')
                         for task, label in zip(tasks, labels)]
            encoded_ = list(zip(tasks, lists, pooled_embs.tolist()))
            encoded += encoded_

    # Output
    if verbose:
        logger.info(f'Write {len(encoded)} embeddings to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        dim = len(encoded[0][-1])
        f.write(f'{len(encoded)} {dim}\n')
        for taskname, listname, emb in encoded:
            emb = ' '.join(map(str, emb))
            if listname is None:
                f.write(f'{taskname.replace(" ", "_")} {emb}\n')
            else:
                f.write(f'{taskname.replace(" ", "_")}@@@{listname.replace(" ", "_")} {emb}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('BaselineTransformers')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--data-type', choices=['LD2018', 'tasks'], required=True, help='Dataset type')
    parser.add_argument('--dummy-list', help='dummy to-do list name for UIT')
    parser.add_argument('--model-type', default='all-mpnet-base.v2', help='SBERT model type')
    parser.add_argument('--concat', choices=['input', 'output'], default='input', help='how to concatenate task and list names. output will double the dimension size')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'max', 'sum'], default='mean', help='how to aggregate token embeddings')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
