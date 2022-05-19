#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from tqdm import tqdm
from spacy.tokens import Doc
import spacy
import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def disable_sent_seg(doc):
    """Disables sentence segmentation"""
    doc[0].is_sent_start = True
    for i in range(1, len(doc)):
        doc[i].is_sent_start = False
    return doc


def main(args):
    global verbose
    verbose = args.verbose

    nlp = spacy.load('en_core_web_lg', exclude=['ner', 'textcat'])
    tok2vec = nlp.get_pipe('tok2vec')
    parser = nlp.get_pipe('parser')
    ruler = nlp.get_pipe('attribute_ruler')

    df = pd.read_csv(args.path_input, delimiter='\t', low_memory=False)
    if verbose:
        logger.info(f'Read {len(df)} rows from {args.path_input}')
    text2tok, text2xpos = {}, {}
    for field in args.field:
        subset = df[[f'{field}.tok', f'{field}.tok']].drop_duplicates()
        text2tok.update({vals[0]: vals[1] for vals in subset.values if isinstance(vals[0], str)})
        subset = df[[f'{field}.tok', f'{field}.xpos']].drop_duplicates()
        text2xpos.update({vals[0]: vals[1] for vals in subset.values if isinstance(vals[0], str)})
    if verbose:
        logger.info(f'{len(text2tok)} unique texts')

    # Parse and write
    if verbose:
        logger.info(f'Write the result to {args.path_output}')
    buff = []
    N = 1000
    with open_helper(args.path_output, 'w') as f:
        for text, text_tok in tqdm(sorted(text2tok.items())):
            tokens = text_tok.split()
            xpos = text2xpos[text].split()
            doc = Doc(nlp.vocab, words=tokens, tags=xpos)
            parsed = ruler(parser(disable_sent_seg(tok2vec(doc))))
            out_text = [
                '\t'.join(map(str, ([i, tok.text, tok.lemma_, tok.pos_, tok.tag_, '_',
                                     tok.head.i+1 if tok.head.i+1 != i else 0,
                                     tok.dep_, '_', '_'])))
                for i, tok in enumerate(parsed, start=1)]
            out_text.append('')
            out_text = [f'# text = {text}'] + out_text
            buff.append('\n'.join(out_text))
            if len(buff) > N:
                f.write('\n'.join(buff) + '\n')
                buff = []
        if len(buff) > 0:
            f.write('\n'.join(buff) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('DepParse')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--field', nargs='+', choices=['task', 'list'],
                        default=['task'], help='text field')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
