#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
import logging
import os
import pickle

import pandas as pd

from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


# Setting a logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s/%(name)s[%(levelname)s]: %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False


class TaskSpecificationDataset(Dataset):
    def __init__(self, filepath: str,
                 tokenizer: PreTrainedTokenizer,
                 use_cache: Optional[bool] = True):
        super(TaskSpecificationDataset, self).__init__()

        path_cache = f'{filepath}.taskspec.cache.pkl'
        if use_cache and os.path.isfile(path_cache):
            logger.info(f'Read {path_cache}')
            with open(path_cache, 'rb') as f:
                self.examples = pickle.load(f)
                return

        df = pd.read_csv(filepath, delimiter='\t', low_memory=False)
        self.examples = []
        for i, (_, row) in enumerate(tqdm(df.iterrows())):
            example = {'id': f'{filepath}###{i}'}
            example['task']= row['task.tok']
            example['task_tokens'] = tokenizer.tokenize(row['task.tok'], is_split_into_words=True)
            example['task_ids'] = LongTensor(
                tokenizer.convert_tokens_to_ids(example['task_tokens']))
            example['list'] = row['list.tok']
            example['list_tokens'] = tokenizer.tokenize(row['list.tok'], is_split_into_words=True)
            example['list_ids'] = LongTensor(
                tokenizer.convert_tokens_to_ids(example['list_tokens']))
            tokens = [tokenizer.cls_token]
            tokens += example['task_tokens']
            tokens.append(tokenizer.sep_token)
            tokens += example['list_tokens']
            tokens.append(tokenizer.eos_token)
            example['input_tokens'] = tokens
            example['input_ids'] = LongTensor(tokenizer.convert_tokens_to_ids(tokens))
            if 'task.full.tok' in row:
                example['task.full'] = row['task.full.tok']
                tokens = [tokenizer.cls_token]
                tokens += tokenizer.tokenize(row['task.full.tok'], is_split_into_words=True)
                tokens.append(tokenizer.eos_token)
                example['output_tokens'] = tokens
                example['output_ids'] = LongTensor(tokenizer.convert_tokens_to_ids(tokens))
            self.examples.append(example)
        logger.info(f'Read {len(self.examples)} examples from {filepath}')

        if use_cache:
            # Save a cache
            logger.info(f'Save a cache to {path_cache}')
            with open(path_cache, 'wb') as f:
                pickle.dump(self.examples, f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
