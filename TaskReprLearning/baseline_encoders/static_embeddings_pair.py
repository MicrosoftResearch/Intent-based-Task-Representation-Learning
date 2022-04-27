#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from typing import Optional
from typing import Tuple
import argparse

from pymagnitude import Magnitude
from pymagnitude import MagnitudeUtils
from tqdm import tqdm
import numpy as np

from baseline_encoders.utils import init_logger
from baseline_encoders.dataset_readers import read_CoTL
from baseline_encoders.utils import batchfy
from baseline_encoders.utils import open_helper

verbose = False

EMBEDDING_MODELS = {
    'word2vec': 'word2vec/medium/GoogleNews-vectors-negative300',
    'fasttext': 'fasttext/medium/crawl-300d-2M.magnitude',
}

def encode(model: Magnitude, input_tokens: List[str]) -> List[np.ndarray]:
    """Encode input tokens and return the last hidden states."""
    return [model.query(tokens) for tokens in input_tokens]

def aggregate_embs(embs, how: str):
    if how == 'mean':  # mean average pooling
        return np.vstack([e.mean(axis=0) for e in embs])
    if how == 'sum':  # sum pooling
        return np.vstack([e.sum(axis=0) for e in embs])
    if how == 'max':  # max pooling
        return np.vstack([e.max(axis=0) for e in embs])

    raise NotImplementedError

def main(args):
    global verbose
    verbose = args.verbose

    # Load a Transformer model
    if verbose:
        logger.info(f'Model type: {args.model_type}')
    if args.model_type in EMBEDDING_MODELS:
        path_model = EMBEDDING_MODELS[args.model_type]
        model = Magnitude(MagnitudeUtils.download_model(path_model))
    else:
        model = Magnitude(args.model_type)

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

    for batch in tqdm(reader_):
        tasks1, lists1, tasks2, lists2, _ = zip(*batch)
        encoded_ = []

        # Tokenization
        for tasks, lists in [(tasks1, lists1), (tasks2, lists2)]:
            input_tokens, input_ids = None, None
            tasks_tok = [text.split() for text in tasks]
            if lists[0] is None:  # No list
                input_tokens = tasks_tok
            else:
                lists_tok = [text.split() for text in lists]
                if args.concat == 'input':
                    input_tokens = tasks_tok + lists_tok

            # Encode
            if input_tokens is not None:
                embs = encode(model, input_tokens)
                pooled_embs = aggregate_embs(embs, how=args.pooling)
            else:
                pooled_embs = []
                for tok in [tasks_tok, lists_tok]:
                    embs = encode(model, tok)
                    pooled_embs.append(aggregate_embs(embs, how=args.pooling))
                pooled_embs = np.hstack(pooled_embs, dim=1)
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
    parser.add_argument('--model-type',
                        required=True, help='Embedding model type')
    parser.add_argument('--concat', choices=['input', 'output'], default='input', help='how to concatenate task and list names. output will double the dimension size')
    parser.add_argument('--pooling', choices=['mean', 'max', 'sum'], default='mean', help='how to aggregate token embeddings')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
