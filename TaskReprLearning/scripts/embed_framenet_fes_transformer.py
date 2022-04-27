#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from typing import Dict
from typing import List
from typing import Optional
import argparse
import logging

from torch import LongTensor
from torch import cat
from torch import nn
from torch import no_grad
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
import numpy as np
import torch

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None

TRANSFORMER_CLASS = {
    ''
}

def encode(model: nn.Module, tokenizer: PreTrainedTokenizer,
           fe: List[str],
           texts: List[str],
           idf: Optional[Dict[str, float]] = None,
           cuda: Optional[int] = -1,
           ) -> List[np.ndarray]:
    """Encode input tokens and return their embeddings."""
    ids, positions = [], []
    for text in texts:
        position = -1
        tokens = text.split()
        for i, token in enumerate(tokens[:len(tokens)-len(fe)+1]):
            for j, fe_ in enumerate(fe):
                if fe_ != tokens[i+j]:
                    break
            else:
                position = i
                break

        # if tokenizer.bos_token is not None:
        #     tokens = [tokenizer.bos_token] + tokens
        #     if position >= 0:
        #         position += 1
        # elif tokenizer.cls_token is not None:
        #     tokens = [tokenizer.cls_token] + tokens
        #     if position >= 0:
        #         position += 1
        # if tokenizer.eos_token is not None:
        #     tokens.append(tokenizer.eos_token)

        ids.append(LongTensor(tokenizer.convert_tokens_to_ids(tokens)))
        positions.append(position)

    ids = pad_sequence(ids,
                       padding_value=tokenizer.pad_token_id,
                       batch_first=True)
    if cuda >= 0:
        ids = ids.to(cuda)
    mask = ids.ne(tokenizer.pad_token_id)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    embs = []
    for i, position in enumerate(positions):
        if position == -1:  # No target FE in the definition -> take mean avg
            e = out.last_hidden_state[i][mask[i]].sum(dim=0, keepdim=True) / mask[i].sum()
            embs.append(e)
            continue
        embs.append(out.last_hidden_state[i, position].unsqueeze(0))
    return cat(embs, dim=0).mean(dim=0).cpu().numpy()

def main(args):
    global verbose
    verbose = args.verbose

    # Load an embedding model
    if verbose:
        logger.info(f'Model type: {args.model_type}')
    if args.model_type.startswith('roberta'):
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)

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
            # Re-tokenization
            fe_ = ' '.join(tokenizer.convert_ids_to_tokens(
                tokenizer(fe.replace('_', ' ').split(),
                          is_split_into_words=True,
                          add_special_tokens=False)['input_ids']))
            defs_tok = [' '.join(tokenizer.convert_ids_to_tokens(
                tokenizer(text.split(),
                          is_split_into_words=True,
                          add_special_tokens=True)['input_ids']))
                        for text in defs_tok]
            data.append((fe, fe_, defs, defs_tok))

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
    model = AutoModel.from_pretrained(args.model_type)
    if args.cuda >= 0:
        model = model.cuda(args.cuda)
    model.eval()
    token_embedder = model.get_input_embeddings()
    embs = []
    for fe, fe_, _, defs_tok in tqdm(data):
        if args.fe_only:
            ids  = LongTensor(tokenizer.convert_tokens_to_ids(fe_.split()))
            if args.cuda >= 0:
                ids = ids.to(args.cuda)
            with no_grad():
                emb = token_embedder(ids).sum(dim=0).cpu().numpy()
        else:
            with no_grad():
                emb = encode(model, tokenizer, fe_.split(), defs_tok, idf=idf, cuda=args.cuda)
        embs.append((fe, ' '.join(map(str, emb))))
    dim = len(emb)

    if verbose:
        logger.info(f'Write {len(embs)} FE embeddings to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write(f'{len(embs)} {dim}\n')
        for fe, emb in sorted(embs):
            f.write(f'{fe} {emb}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('EmbedFE')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('-m', '--model-type',
                        required=True, help='transformer model type')
    parser.add_argument('--fe-only', action='store_true', help='only use FE text')
    parser.add_argument('--lowercase', action='store_true', help='lowercase text')
    parser.add_argument('--tfidf', action='store_true', help='use TFIDF weighting')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('--cuda', type=int, default=-1, help='GPU device number')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
