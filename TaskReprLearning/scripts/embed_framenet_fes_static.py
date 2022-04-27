#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from typing import Dict
from typing import List
from typing import Optional
import argparse
import logging

from pymagnitude import Magnitude
from pymagnitude import MagnitudeUtils
from tqdm import tqdm
import numpy as np

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None

EMBEDDING_MODELS = {
    'word2vec': 'word2vec/medium/GoogleNews-vectors-negative300',
    'fasttext': 'fasttext/medium/crawl-300d-2M.magnitude',
}

def encode(model: Magnitude, texts: List[str],
           idf: Optional[Dict[str, float]] = None
           ) -> List[np.ndarray]:
    """Encode input tokens and return their embeddings."""
    embs = [model.query(text.split(' ')) for text in texts]

    if idf is None:
        # 1: Avg with uniform weights
        ## mean average per sentence -> mean average over all sentences
        emb = np.vstack([e.mean(axis=0) for e in embs]).mean(axis=0)
    else:
        # 2: Avg with TF-IDF weights
        tf, N = defaultdict(int), 0
        for token in chain.from_iterable([text.split(' ') for text in texts]):
            tf[token] += 1
            N += 1
        tfidf = [np.array([tf[tok]*idf[tok] for tok in text.split(' ')]) for text in texts]
        emb = np.vstack([(e * w.reshape((-1, 1))).sum(axis=0) for e, w in zip(embs, tfidf)]).mean(axis=0)
    return emb

def main(args):
    global verbose
    verbose = args.verbose

    # Load an embedding model
    if verbose:
        logger.info(f'Model type: {args.model_type}')
    path_model = EMBEDDING_MODELS[args.model_type]
    model = Magnitude(MagnitudeUtils.download_model(path_model))

    # Read the input file
    ## Format (TAB-delimited)
    ## <FE> <definitions> <tokenized definitions>
    if verbose:
        logger.info(f'Read {args.path_input}')
        logger.info(f'Lowercase: {args.lowercase}')
    data = []
    with open_helper(args.path_input) as f:
        for line in f:
            fe, defs, defs_tok = line.strip().split('\t')
            if args.lowercase:
                defs = defs.lower()
                defs_tok = defs_tok.lower()
            defs = defs.split('@@@')
            defs_tok = defs_tok.split('@@@')
            data.append((fe, defs, defs_tok))

    # Compute TFIDF weights
    idf = None
    if args.tfidf:
        if verbose:
            logger.info('Use TF-IDF weighting')
        df = defaultdict(int)
        for fe, _, defs_tok in data:
            tokens = []
            for sent in defs_tok:
                tokens += sent.split(' ')
            for tok in set(tokens):
                df[tok] += 1
        idf = {}
        for tok, counts in df.items():
            idf[tok] = np.log((len(data)+1)/(counts+1)) + 1  # smoothing
            # Assume one additional document containing all terms
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer

    # Encode
    embs = []
    for fe, _, defs_tok in tqdm(data):
        emb = encode(model, defs_tok, idf=idf)
        embs.append((fe, ' '.join(map(str, emb))))

    if verbose:
        logger.info(f'Write {len(embs)} FE embeddings to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write(f'{len(embs)} {model.dim}\n')
        for fe, emb in sorted(embs):
            f.write(f'{fe} {emb}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('EmbedFE')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('-m', '--model-type', choices=['word2vec', 'fasttext'],
                        required=True, help='Embedding model type')
    parser.add_argument('--lowercase', action='store_true', help='lowercase text')
    parser.add_argument('--tfidf', action='store_true', help='use TFIDF weighting')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
