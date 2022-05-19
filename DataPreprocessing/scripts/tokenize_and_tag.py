#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

import spacy

nlp = spacy.load('en_core_web_lg', disable=['parse', 'ner', 'textcat'])

r_space = re.compile(r'\s\s+')

buff = []
for line in sys.stdin:
    if len(line.strip()) == 0:
        continue
    doc = nlp(r_space.sub(' ', line.strip()))
    token = ' '.join(tok.text for tok in doc)
    lemma = ' '.join(tok.lemma_ for tok in doc)
    upos = ' '.join(tok.pos_ for tok in doc)
    xpos = ' '.join(tok.tag_ for tok in doc)
    buff.append('\t'.join([line.strip(), token, lemma, upos, xpos]))
    if len(buff) > 10000:
        print('\n'.join(buff))
        buff = []

if len(buff) > 0:
    print('\n'.join(buff))
