#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from spacy.tokens import Doc
from tqdm import tqdm
import spacy

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False
logger = None


def extract_pas(doc):
    sents = list(doc.sents)
    assert len(sents) == 1
    sent = sents[0]

    root = sent.root
    noun_chunks = {chunk.root.i: chunk for chunk in doc.noun_chunks}
    def get_noun_chunk(token):
        return noun_chunks.get(token.i, list(token.subtree))

    predicate, arguments = [], {}
    if root.tag_ == 'VB':  # VP
        predicate = [(root.i, root)]
        for child in root.children:
            dep = child.dep_
            if dep in {'compound', 'prt'} and len(list(child.children)) == 0:
                predicate.append((child.i, child))
                continue
            if dep in {'aux', 'neg'}:
                predicate.append((child.i, child))
            if dep in {'attr', 'dobj', 'pobj'}:  # attr for copula verbs
                # pobj --> object of a preposition (probably parsing error)
                arguments[dep] = []
                arguments[dep].append(get_noun_chunk(child))
                for conjunct in child.conjuncts:
                    arguments[dep].append(get_noun_chunk(conjunct))
                continue
            if dep == 'nsubj':
                arguments['nsubj'] = [get_noun_chunk(child)]
                for conjunct in child.conjuncts:
                    arguments['nsubj'].append(get_noun_chunk(conjunct))
                continue
            if dep == 'csubj':
                arguments['csubj'] = [list(child.subtree)]
                continue
            if dep == 'dative':
                arguments['dative'] = [get_noun_chunk(child)]
                for conjunct in child.conjuncts:
                    arguments['dative'].append(get_noun_chunk(conjunct))
                continue
            if dep == 'oprd':  # object predicate - She calls [me] [herfridn] -> oprd(calls, friend)
                arguments['oprd'] = [get_noun_chunk(child)]
                for conjunct in child.conjuncts:
                    arguments['oprd'].append(get_noun_chunk(conjunct))
                continue
            if dep in {'acl', 'amod', 'advcl', 'advmod', 'npadvmod', 'acomp', 'ccomp', 'xcomp', 'mark', 'relcl'}:
                if dep not in arguments:
                    arguments[dep] = []
                arguments[dep].append(list(child.subtree))
                continue
            if dep == 'prt':  # particle which has a child (probably parsing error)
                if 'prt' not in arguments:
                    arguments['prt'] = []
                arguments['prt'].append(list(child.subtree))
                continue
            if dep == 'prep':  # 'prepositional modifier'
                if 'prep' not in arguments:
                    arguments['prep'] = []
                arguments['prep'].append(list(child.subtree))
                continue
            # NP-related modifiers -- probably noise
            if dep in {'nmod', 'appos', 'det', 'nummod', 'poss', 'predet'}:
                continue
            if dep in {'dep'}:  # probably noise
                predicate += [(elm.i, elm) for elm in child.subtree]
                continue
            if dep in {'attr', 'aux', 'case', 'compound', 'intj', 'meta', 'parataxis', 'punct'}:
                continue
            if dep in {'cc', 'conj'}:
                # Note: perhaps we should just skip tasks with conjunction - too complicated
                continue
        _, predicate = zip(*sorted(predicate))
    elif root.pos_ in {'NOUN', 'PRON', 'PROPN'}:  # NP
        arguments['root'] = [get_noun_chunk(root)]

    predicate = ' '.join(tok.text for tok in predicate)
    for key, vals in arguments.items():
        arguments[key] = [' '.join(tok.text for tok in val) for val in vals]

    return predicate, arguments


def main(args):
    global verbose
    verbose = args.verbose

    nlp = spacy.load('en_core_web_lg', exclude=['ner', 'textcat'])

    if verbose:
        logger.info(f'Read {args.path_input}')
    data = []
    with open_helper(args.path_input) as f:
        buff, text = [], None
        for i, line in tqdm(enumerate(f)):
            line = line.strip('\n')
            if len(line) == 0:
                # Process buffer
                _, tokens, lemmas, pos, tags, _, heads, deps, _, _ = zip(*buff)
                heads = [None if v == '0' else int(v)-1 for v in heads]
                sent_starts = [True if i == 0 else False for i, _ in enumerate(tokens)]
                doc = Doc(nlp.vocab, words=tokens, lemmas=lemmas,
                          pos=pos, tags=tags, heads=heads, deps=deps,
                          sent_starts=sent_starts)
                predicate, arguments = extract_pas(doc)
                data.append({
                    'text': text,
                    'predicate': predicate,
                    'arguments': arguments
                    })
                if len(data) == 10000:
                    with open('memo.json', 'w') as f:
                        f.write('\n'.join(json.dumps(record) for record in data) + '\n')
                buff = []
                continue
            if line.startswith('# text'):
                text = line[9:]
                continue
            buff.append(line.split('\t'))

    if verbose:
        logger.info(f'Write {len(data)} to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write('\n'.join(json.dumps(record) for record in data) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('ExtPA')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
