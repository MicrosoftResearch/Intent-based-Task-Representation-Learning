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
import torch
import pandas as pd

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


class TasksDataset(Dataset):
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
                 default_list: Optional[str] = 'inbox',
                 no_list: Optional[bool] = False,
                 **kwargs):
        super(TasksDataset, self).__init__()

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
        path_cache = f'{filepath}.tasks.cache.pkl'
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

        # Input format (TAB-delimited: [task]<TAB>[list]):
        #   buy milk	inbox
        #   milk	groceries
        #   call #OOV#	today
        #   go to #OOV#	today
        self.examples = []

        logger.info(f'Read {filepath}')
        with open_helper(filepath) as f:
            if local_rank in [-1, 0]:
                iterator = tqdm(enumerate(f), dynamic_ncols=True)
            else:
                iterator = enumerate(f)
            logger.info(f'Default list name: {default_list}')
            default_list_tok = tokenizer.tokenize(default_list, is_split_into_words=True)
            default_list_ids = LongTensor(tokenizer.convert_tokens_to_ids(default_list_tok))
            for i, line in iterator:
                example = {'id': f'{filepath}###{i}'}
                rows = line.strip().split('\t')
                for field, text in zip(['task', 'list'], rows):
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
                    example[field] = text
                    example[f'{field}_tokens'] = tokens
                    example[f'{field}_ids'] = LongTensor(ids)
                if 'list' not in example:  # list name is not provided in the file
                    example['list'] = default_list
                    example['list_tokens'] = default_list_tok
                    example['list_ids'] = default_list_ids

                # Input
                tokens = [tokenizer.bos_token] if add_bos_to_input else []
                token_types = [0] if add_bos_to_input else []
                tokens += example['task_tokens']
                if no_other_type_id:
                    token_types += [0 for _ in range(len(example['task_tokens']))]
                else:
                    token_types += [1 for _ in range(len(example['task_tokens']))]
                if not no_list:
                    if add_sep_to_input:
                        tokens.append(tokenizer.sep_token)
                        token_types.append(0)
                    tokens += example['list_tokens']
                    if no_other_type_id:
                        token_types += [1 for _ in range(len(example['list_tokens']))]
                    else:
                        token_types += [2 for _ in range(len(example['list_tokens']))]
                if add_eos_to_input:
                    tokens.append(tokenizer.eos_token)
                    if no_other_type_id:
                        token_types.append(1)
                    else:
                        token_types.append(0)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                example['input_tokens'] = tokens
                example['input_ids'] = LongTensor(ids)
                example['input_type_ids'] = LongTensor(np.array(token_types))

                if len(rows) > 2:
                    example['label'] = '\t'.join(rows[2:])

                self.examples.append(example)

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
