#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy
from typing import Dict
from typing import Optional
import json
import logging
import os
import pickle
import random

from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import numpy as np
import torch

from src.utils import open_helper


# Setting a logger
default_logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s/%(name)s[%(levelname)s]: %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(handler)
default_logger.propagate = False


class MultiTaskDataset(Dataset):
    def __init__(self, filepath: str,
                 tokenizer: PreTrainedTokenizer,
                 local_rank: Optional[int] = -1,
                 use_cache: Optional[bool] = True,
                 overwrite_cache: Optional[bool] = False,
                 is_validation: Optional[bool] = True,
                 logger: Optional[logging.Logger] = None,
                 add_bos_to_input: Optional[bool] = True,
                 add_eos_to_input: Optional[bool] = True,
                 add_sep_to_input: Optional[bool] = True,
                 no_other_type_id: Optional[bool] = False,
                 no_list: Optional[bool] = False,
                 dummy_list: Optional[bool] = False,
                 use_sep_as_eos: Optional[bool] = False,
                 lowercase: Optional[bool] = False,
                 aux_label_indexers: Optional[Dict[str, Dict[str, int]]] = {},
                 framenet_label_path: Optional[str] = None,
                 **kwargs):
        super(MultiTaskDataset, self).__init__()

        self.is_validation = is_validation  # do not randomize instances when is_validation = T

        if logger is None:
            logger = default_logger

        # Read label indexer for framenet
        self.aux_label_indexers = aux_label_indexers
        if 'framenet' not in self.aux_label_indexers and framenet_label_path is not None:
            self.aux_label_indexers['framenet'] = {tokenizer.pad_token: 0,
                                                   tokenizer.unk_token: 1}
            with open_helper(framenet_label_path) as f:
                next(f)  # skip a header
                for i, line in enumerate(f, start=2):
                    label, _ = line.split(' ', 1)
                    self.aux_label_indexers['framenet'][label] = i

        tokenizer_name = tokenizer.name_or_path.replace('/', '__')
        tags = [tokenizer_name]
        if no_list:
            tags.append('no-list')
        if dummy_list:
            tags.append('dummy-list')
        if not add_bos_to_input:
            tags.append('no-bos')
        if not add_eos_to_input:
            tags.append('no-eos')
        if not add_sep_to_input:
            tags.append('no-sep')
        if no_other_type_id:
            tags.append('no-other-type')
        if use_sep_as_eos:
            tags.append('sep2eos')
            # This automatically updates tokenizer.eos_token_id, too.
            tokenizer.eos_token = tokenizer.sep_token
        if lowercase:
            tags.append('lowercase')
        if framenet_label_path:
            tags.append('fn-emb')
        tags = '.'.join(tags)
        path_cache = f'{filepath}.{tags}.multitask.cache.pkl'
        cache_exists = os.path.isfile(path_cache)
        if local_rank not in [-1, 0]:
            if use_cache and not cache_exists:
                torch.distributed.barrier()  # wait for proc 0 creating a cache
                cache_exists = os.path.isfile(path_cache)  # re-check
                assert cache_exists
        if use_cache and (not overwrite_cache) and cache_exists:
            logger.info(f'Read {path_cache}')
            with open(path_cache, 'rb') as f:
                self.examples = pickle.load(f)
                # sanity check
                logger.info(self.examples[0])
                logger.info(self.examples[42])
                return

        logger.info(f'Read {filepath}')
        logger.info(f'Tags: {tags}')
        logger.info(f'Overwrite cache?: {overwrite_cache}')
        self.examples = []
        with open_helper(filepath) as f:
            for i, line in tqdm(enumerate(f)):
                row = json.loads(line)
                example = {'id': f'{filepath}###{i}'}
                _tokens = {}
                for field in ['task', 'list']:
                    example[field] = row[f'{field}.tok']
                    text = example[field]
                    if field == 'list' and dummy_list:
                        text = 'inbox'
                    if lowercase:
                        text = text.lower()
                        # Replace #OOV# with UNK
                        text = text.replace('#oov#', tokenizer.unk_token)
                    else:
                        text = text.replace('#OOV#', tokenizer.unk_token)
                    tokenized = tokenizer(text.split(),
                                          is_split_into_words=True,
                                          add_special_tokens=False)
                    ids = tokenized['input_ids']
                    tokens = tokenizer.convert_ids_to_tokens(ids)
                    example[f'{field}_ids'] = LongTensor(ids)
                    # example[f'{field}_tokens'] = tokens
                    _tokens[field] = tokens

                # Input
                ## Token type (0: special symbols, 1: task, 2: list)
                ## Token type (0: CLS+task+SEP, 1: list+SEP) (when no_other_type_id is True)
                tokens = [tokenizer.bos_token] if add_bos_to_input else []
                token_types = [0] if add_bos_to_input else []
                tokens += _tokens['task']
                # tokens += example['task_tokens']
                if no_other_type_id:
                    token_types += [0 for _ in range(len(_tokens['task']))]
                    # token_types += [0 for _ in range(len(example['task_tokens']))]
                else:
                    token_types += [1 for _ in range(len(_tokens['task']))]
                    # token_types += [1 for _ in range(len(example['task_tokens']))]
                if not no_list:
                    if add_sep_to_input:
                        tokens.append(tokenizer.sep_token)
                        token_types.append(0)
                    tokens += _tokens['list']
                    # tokens += example['list_tokens']
                    if no_other_type_id:
                        token_types += [1 for _ in range(len(_tokens['list']))]
                        # token_types += [1 for _ in range(len(example['list_tokens']))]
                    else:
                        token_types += [2 for _ in range(len(_tokens['list']))]
                        # token_types += [2 for _ in range(len(example['list_tokens']))]
                if add_eos_to_input:
                    tokens.append(tokenizer.eos_token)
                    if no_other_type_id:
                        token_types.append(1)
                    else:
                        token_types.append(0)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                # example['input_tokens'] = tokens
                example['input_ids'] = LongTensor(ids)
                example['input_type_ids'] = LongTensor(np.array(token_types))

                # Output
                if 'task.full.tok' in row:
                    if isinstance(row['task.full.tok'], str):
                        row['task.full.tok'] = [row['task.full.tok']]
                    example['task.full'] = row['task.full.tok']
                    example['output_tokens'] = []
                    example['output_ids'] = []
                    for text in row['task.full.tok']:
                        if lowercase:
                            text = text.lower()
                            text = text.replace('#oov#', tokenizer.unk_token)
                        else:
                            text = text.replace('#OOV#', tokenizer.unk_token)
                        tokenized = tokenizer(text.split(),
                                              is_split_into_words=True,
                                              add_special_tokens=False)

                        ids = [tokenizer.bos_token_id] + tokenized['input_ids'] + [tokenizer.eos_token_id]
                        tokens = tokenizer.convert_ids_to_tokens(ids)
                        example['output_ids'].append(LongTensor(ids))
                        # example['output_tokens'].append(tokens)

                for col, texts in row.items():
                    if col not in {'comet-xNeed', 'comet-xIntent', 'autosuggest'}:
                        continue
                    if len(texts) == 0:
                        continue
                    if isinstance(texts[0], str):
                        texts = [texts]
                    if lowercase:
                        texts = [[text.lower() for text in texts_] for texts_ in texts]
                    example[col] = texts
                    example[f'{col}_tokens'] = []
                    example[f'{col}_ids'] = []
                    for texts_ in texts:
                        _tokens = [
                            tokenizer.tokenize(text, is_split_into_words=True)
                            for text in texts_
                        ]
                        if col.startswith('comet'):
                            _tokens = [
                                [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
                                for tokens in _tokens[:]]
                        # example[f'{col}_tokens'][-1] = _tokens
                        example[f'{col}_ids'].append([
                            LongTensor(tokenizer.convert_tokens_to_ids(tokens))
                            for tokens in _tokens
                        ])
                if isinstance(row.get('framenet', None), list):
                    texts, core_texts = [], []
                    for framenet in row['framenet']:
                        if not isinstance(framenet, dict):
                            texts.append('')
                            core_texts.append('')
                            continue
                        texts_ = []
                        for text in set(framenet['core'] + framenet['noncore']):
                            if not isinstance(text, str):
                                continue
                            texts_.append(text.split('@@@')[0])
                        assert len(texts_) == len(set(texts_))
                        texts.append(' '.join(texts_))
                        texts_ = []
                        for text in set(framenet['core']):
                            if not isinstance(text, str):
                                continue
                            texts_.append(text.split('@@@')[0])
                        assert len(texts_) == len(set(texts_))
                        core_texts.append(' '.join(texts_))
                    example['framenet'] = texts
                    if 'framenet' in self.aux_label_indexers:
                        # example['framenet_tokens'] = texts
                        example['framenet_ids'] = [
                            None if labels is None
                            else LongTensor(sorted([self.aux_label_indexers['framenet'][label]
                                                    for label in labels.split()]))
                            for labels in texts]
                        # example['framenet_core_tokens'] = core_texts
                        example['framenet_core_ids'] = [
                            None if labels is None
                            else LongTensor(sorted([self.aux_label_indexers['framenet'][label]
                                                    for label in labels.split()]))
                            for labels in core_texts]
                    else:
                        # Use Transformer's tokenizer
                        if lowercase:
                            texts = [text.lower() for text in texts]
                        example['framenet_tokens'] = [
                            None if text is None else tokenizer.tokenize(
                                text.replace('_', ' ').split(), is_split_into_words=True)
                            for text in texts]
                        example['framenet_ids'] = [
                            None if tokens is None else LongTensor(tokenizer.convert_tokens_to_ids(tokens))
                            for tokens in example['framenet_tokens']]

                        if lowercase:
                            core_texts = [text.lower() for text in core_texts]
                        example['framenet_core_tokens'] = [
                            None if text is None else tokenizer.tokenize(
                                text.replace('_', ' ').split(), is_split_into_words=True)
                            for text in core_texts]
                        example['framenet_core_ids'] = [
                            None if tokens is None else LongTensor(tokenizer.convert_tokens_to_ids(tokens))
                            for tokens in example['framenet_core_tokens']]

                self.examples.append(example)
                if i in [0, 42]:  # sanity check
                    logger.info(example)

        if use_cache and local_rank in [-1, 0]:
            # Save a cache
            logger.info(f'Save a cache to {path_cache}')
            with open(path_cache, 'wb') as f:
                pickle.dump(self.examples, f)

            if local_rank == 0 and not cache_exists:
                torch.distributed.barrier()  # resume the other processes

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = copy(self.examples[index])

        output_i = None
        if 'output_ids' in example:
            if self.is_validation:
                i = 0
            else:
                i = random.randint(1, len(example['output_ids'])) - 1
            example['output_ids'] = example['output_ids'][i]
            # example['output_tokens'] = example['output_tokens'][i]
            output_i = i

        if output_i is None:
            output_i = 0

        if 'framenet_ids' in example:
            example['framenet_ids'] = example['framenet_ids'][output_i]
            # example['framenet_tokens'] = example['framenet_tokens'][output_i]
            example['framenet_core_ids'] = example['framenet_core_ids'][output_i]
            # example['framenet_core_tokens'] = example['framenet_core_tokens'][output_i]
            if example['framenet_ids'] is None or len(example['framenet_ids']) == 0:
                del example['framenet_ids']
                # del example['framenet_tokens']
                del example['framenet_core_ids']
                # del example['framenet_core_tokens']
        for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest']:
            if f'{key}_ids' in example and len(example[f'{key}_ids'][output_i]) > 0:
                if self.is_validation:
                    i = 0
                else:
                    i = random.randint(1, len(example[f'{key}_ids'][output_i])) - 1
                if len(example[f'{key}_ids'][output_i][i]) == 0:
                    del example[f'{key}_ids']
                    # del example[f'{key}_tokens']
                    continue
                example[f'{key}_ids'] = example[f'{key}_ids'][output_i][i]
                # example[f'{key}_tokens'] = example[f'{key}_tokens'][output_i][i]
            else:
                try:
                    del example[f'{key}_ids']
                    # del example[f'{key}_tokens']
                except KeyError:
                    pass

        return example
