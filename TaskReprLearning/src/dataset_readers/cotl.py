#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any
from typing import Dict
from typing import Optional
import logging
import os
import pickle

from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import numpy as np
import pandas as pd
import torch


# Setting a logger
default_logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s/%(name)s[%(levelname)s]: %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(handler)
default_logger.propagate = False


class CoTLDataset(Dataset):
    """Co-time and co-location task pair prediction"""
    def __init__(self, filepath: str,
                 tokenizer: PreTrainedTokenizer,
                 local_rank: Optional[int] = -1,
                 use_cache: Optional[bool] = True,
                 logger: Optional[logging.Logger] = None,
                 add_bos_to_input: Optional[bool] = True,
                 add_eos_to_input: Optional[bool] = True,
                 add_sep_to_input: Optional[bool] = True,
                 no_other_type_id: Optional[bool] = False,
                 lowercase: Optional[bool] = False,
                 no_list: Optional[bool] = False,
                 **kwargs):
        super(CoTLDataset, self).__init__()

        if logger is None:
            logger = default_logger
        tokenizer_name = tokenizer.name_or_path.replace('/', '__')
        tags = [tokenizer_name]
        if not add_bos_to_input:
            tags.append('no-bos')
        if not add_eos_to_input:
            tags.append('no-eos')
        if not add_sep_to_input:
            tags.append('no-sep')
        if no_other_type_id:
            tags.append('no-other-type')
        if lowercase:
            tags.append('lowercase')
        tags = '.'.join(tags)
        path_cache = f'{filepath}.{tags}.cotl.cache.pkl'
        cache_exists = os.path.isfile(path_cache)
        if local_rank not in [-1, 0]:
            if use_cache and not cache_exists:
                torch.distributed.barrier()  # wait for proc 0 creating a cache
                cache_exists = os.path.isfile(path_cache)  # re-check
                assert cache_exists
        if use_cache and cache_exists:
            logger.info(f'Read {path_cache}')
            with open(path_cache, 'rb') as f:
                self.examples = pickle.load(f)
                return

        # Input format:
        # Label(True|False)<TAB>TaskTitle1<TAB>ListTitle1<TAB>TaskTitle2<TAB>ListTitle2
        self.examples = []
        df = pd.read_csv(filepath, delimiter='\t')
        logger.info(f'Read {len(df)} rows from {filepath}')
        if local_rank in [-1, 0]:
            iterator = tqdm(df.iterrows(), dynamic_ncols=True)
        else:
            iterator = df.iterrows()
        for idx, row in iterator:
            example = {'id': f'{filepath}###{idx}'}
            for i in [1, 2]:
                example[f'task{i}'] = row[f'TaskTitle{i}']
                text = example[f'task{i}']
                text = text.replace('#OOV#', tokenizer.unk_token)
                if lowercase:
                    text = text.lower()
                tokenized = tokenizer(text.split(),
                  is_split_into_words=True,
                  add_special_tokens=False)
                ids = tokenized['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(ids)
                example[f'task{i}_tokens'] = tokens
                example[f'task{i}_ids'] = LongTensor(ids)

                example[f'list{i}'] = row[f'ListTitle{i}']
                if example[f'list{i}'] == 'default list':
                    example[f'list{i}'] = 'inbox'
                text = example[f'list{i}']
                if lowercase:
                    text = text.lower()
                    text = text.replace('#oov#', tokenizer.unk_token)
                else:
                    text = text.replace('#OOV#', tokenizer.unk_token)
                tokenized = tokenizer(text.split(),
                  is_split_into_words=True,
                  add_special_tokens=False)
                ids = tokenized['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(ids)
                example[f'list{i}_tokens'] = tokens
                example[f'list{i}_ids'] = LongTensor(ids)

                # Input
                ## Token type (0: special symbols, 1: task, 2: list)
                tokens = [tokenizer.bos_token] if add_bos_to_input else []
                token_types = [0] if add_bos_to_input else []
                tokens += example[f'task{i}_tokens']
                if no_other_type_id:
                    token_types += [0 for _ in range(len(example[f'task{i}_tokens']))]
                else:
                    token_types += [1 for _ in range(len(example[f'task{i}_tokens']))]
                if not no_list:
                    if add_sep_to_input:
                        tokens.append(tokenizer.sep_token)
                        token_types.append(0)
                    tokens += example[f'list{i}_tokens']
                    if no_other_type_id:
                        token_types += [1 for _ in range(len(example[f'list{i}_tokens']))]
                    else:
                        token_types += [2 for _ in range(len(example[f'list{i}_tokens']))]
                if add_eos_to_input:
                    tokens.append(tokenizer.eos_token)
                    if no_other_type_id:
                        token_types.append(1)
                    else:
                        token_types.append(0)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                example[f'input_tokens{i}'] = tokens
                example[f'input_ids{i}'] = LongTensor(ids)
                example[f'input_type_ids{i}'] = LongTensor(np.array(token_types))

            if 'Label' in example:
                example['label'] = row['Label'].strip()
                example['label_id'] = int(example['label'].lower() == 'true')

            self.examples.append(example)
            if idx in [0, 42]:  # sanity check
                logger.info(example)

        if use_cache and local_rank in [-1, 0]:
            # Save a cache
            logger.info(f'Save a cache to {path_cache}')
            with open(path_cache, 'wb') as f:
                pickle.dump(self.examples, f)

            if local_rank == 0 and not cache_exists:
                torch.distributed.barrier()  # resume the other processes

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]
