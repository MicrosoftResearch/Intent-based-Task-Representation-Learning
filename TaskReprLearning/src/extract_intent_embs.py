#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
from os import path
import argparse
import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig
from transformers import BertModel
from transformers import BertTokenizer
from transformers import GPT2Model
from transformers import GPT2Tokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizer
import torch

from src.dataset_readers import LD2018Dataset
from src.dataset_readers import MultiTaskDataset
from src.dataset_readers import TasksDataset
from src.models import SimpleCGenClfModel
from src.models import SimpleGenClfModel
from src.utils import init_logger
from src.utils import open_helper
from src.utils import read_config

TOKENIZER_CLASS = {
    'bert-base-cased': BertTokenizer,
    'bert-base-uncased': BertTokenizer,
    'bert-large-cased': BertTokenizer,
    'gpt2': GPT2Tokenizer,
    'roberta-base': RobertaTokenizer,
}
TRANSFORMER_CLASS = {
    'bert-base-cased#continuous': BertModel,
    'bert-base-cased#discrete': BertModel,
    'bert-large-cased#continuous': BertModel,
    'bert-large-cased#discrete': BertModel,
    'bert-base-uncased#continuous': BertModel,
    'bert-base-uncased#discrete': BertModel,
    'bert-large-uncased#continuous': BertModel,
    'bert-large-uncased#discrete': BertModel,
    'gpt2#continuous': GPT2Model,
    'gpt2#discrete': GPT2Model,
    'roberta-base#continuous': RobertaModel,
    'roberta-base#discrete': RobertaModel,
}

DATASET_CLASS = {
    'LD2018': LD2018Dataset,
    'multitask': MultiTaskDataset,
    'tasks': TasksDataset,
}

verbose = False


def main(args):
    global verbose
    verbose = args.verbose

    config = read_config(path.join(args.model_dir, 'config.json'))
    model_config = config.get('model', {})
    transformer_config = model_config.get('transformer', {})

    model_type = transformer_config.get('type')
    tokenizer_cls = TOKENIZER_CLASS[model_type]
    primary_output = model_config.get('primary_output', 'discrete')
    transformer_cls = TRANSFORMER_CLASS[model_type + '#' + primary_output]

    tokenizer = tokenizer_cls.from_pretrained(path.join(args.model_dir, 'tokenizer'))
    kwargs = transformer_config.get('params', {})
    transformer = transformer_cls.from_pretrained(model_type, **kwargs)
    if transformer.config.vocab_size != len(tokenizer):
        transformer.resize_token_embeddings(len(tokenizer))
    wrapper_model_type = model_config.get('type', 'simple_encoder')
    if wrapper_model_type == 'simple_encoder':
        if primary_output == 'discrete':
            model = SimpleGenClfModel(transformer, tokenizer, **model_config)
        else:
            model = SimpleCGenClfModel(transformer, tokenizer, **model_config)
    else:
        raise NotImplementedError
    path_model = path.join(args.model_dir, 'model.pth' if args.use_final else 'model.best.pth')
    logger.info(f'Load {path_model}')
    # Adhoc fix (why do we need this?)
    state_dict = torch.load(path_model, map_location='cpu')
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        new_state_dict[key] = val
    model.load_state_dict(new_state_dict)

    model.eval()

    logger.info(f'Read {args.path_input} (type: {args.data_type})')
    dataset_config = config['dataset_reader'][0]
    del dataset_config['filepath']
    dataset_cls = DATASET_CLASS[args.data_type]
    if dataset_config.get('no_list', False):
        args.no_list = True
        del dataset_config['no_list']
    dataset = dataset_cls(args.path_input, tokenizer, use_cache=False, no_list=args.no_list,
                          **dataset_config)

    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    def collate(examples):
        batch = {}
        cols = set(chain.from_iterable([ex.keys() for ex in examples]))
        seqs = [ex['input_ids'] for ex in examples]

        if 'input_type_ids' in cols:
            type_seqs = [ex['input_type_ids'] for ex in examples]
            batch['input_type_ids'] = pad_sequence(type_seqs,
                                                   padding_value=pad_token_id,
                                                   batch_first=True)
        else:
            # Make token type IDs
            token_types = []
            for seq in seqs:
                ids = torch.zeros_like(seq)
                sep_i = seq.eq(sep_token_id).nonzero().flatten()[0]
                ids[1:sep_i] = 1  # task
                ids[sep_i+1:-1] = 2  # list
                token_types.append(ids)
            batch['input_type_ids'] = pad_sequence(token_types,
                                                   padding_value=0,
                                                   batch_first=True)
        batch['input_ids'] = pad_sequence(seqs, padding_value=pad_token_id,
                                          batch_first=True)
        batch['input_mask'] = batch['input_ids'].ne(pad_token_id)
        batch['_input_tokens'] = [ex['input_tokens'] for ex in examples]
        batch['_task'] = [ex['task'] for ex in examples]
        batch['_list'] = [ex['list'] for ex in examples]
        batch['_label'] = [ex.get('label') for ex in examples]
        batch['_task.full'] = [ex.get('task.full', '') for ex in examples]

        return batch

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=args.batch_size,
                            collate_fn=collate, drop_last=False)

    if args.cuda >= 0:
        model = model.to(args.cuda)

    buff = {col: [] for col in ['task', 'list', 'emb',
                                'tokens', 'attn', 'attn_norm', 'attn_accum']}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.cuda >= 0:
                for key, val in batch.items():
                    if key.startswith('_'):
                        continue
                    batch[key] = val.to(args.cuda)
            if args.no_extractor:
                try:
                    encoder = model.transformer.get_encoder()
                    out = encoder(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['input_mask'])
                except AttributeError:
                    out = model.transformer(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['input_mask'])
                h = out['last_hidden_state']
                # batch['input_mask'] = (batch['input_mask'] & (batch['input_type_ids'] != 2))
                h[~batch['input_mask']] = 0
                if args.pooling == 'mean':
                    intent_embs = h.sum(dim=1) / batch['input_mask'].sum(dim=1, keepdim=True)
                else:
                    raise NotImplementedError
                attns, attn_norms = None, None
            else:
                intent_embs, attns, attn_norms, attn_accum = model.get_intent_embs(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['input_mask'],
                    input_type_ids=batch.get('input_type_ids'),
                    return_attention_scores=True)
            if intent_embs.device != 'cpu':
                intent_embs = intent_embs.cpu()
                if attns is not None:
                    attns = attns.cpu()
                    attn_norms = attn_norms.cpu()
                    attn_accum = attn_accum.cpu()

            for i, (taskname, listname, taskname_full, tokens) in enumerate(
                    zip(batch['_task'], batch['_list'], batch['_task.full'],
                        batch['_input_tokens'])):
                buff['task'].append(taskname.replace(' ', '_'))
                buff['list'].append(listname.replace(' ', '_'))
                buff['emb'].append(' '.join(map(str, intent_embs[i].numpy().tolist())))
                if attns is not None:
                    buff['tokens'].append(tokens)
                    buff['attn'].append(attns[i].numpy().tolist()[:len(tokens)])
                    buff['attn_norm'].append(attn_norms[i].numpy().tolist()[:len(tokens)])
                    buff['attn_accum'].append(attn_accum[i].numpy().tolist()[:len(tokens)])

    logger.info(f'Write to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write(f'{len(buff["task"])} {len(buff["emb"][0].split())}\n')
        for taskname, listname, emb in zip(buff['task'], buff['list'], buff['emb']):
            f.write(f'{taskname}@@@{listname} {emb}\n')

    if len(buff['attn']) > 0:
        path_output = args.path_output + '.attn.json'
        if verbose:
            logger.info(f'Write attention scores to {path_output}')
        with open(path_output, 'w') as f:
            for record in zip(buff['task'], buff['list'],
                              buff['tokens'],
                              buff['attn'], buff['attn_norm'], buff['attn_accum']):
                f.write(json.dumps({
                    'task': record[0],
                    'list': record[1],
                    'tokens': record[2],
                    'attn': record[3],
                    'attn_norm': record[4],
                    'attn_accum': record[5]
                }) + '\n')

    return 0

if __name__ == '__main__':
    logger = init_logger('Generation')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to a input file')
    parser.add_argument('--data-type', choices=['multitask', 'LD2018', 'tasks'], default='multitask', help='Dataset type')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to output file')
    parser.add_argument('-m', '--model-dir', type=str, required=True, help='model directory')
    parser.add_argument('--use-final', action='store_true', default=False, help='use the final checkpoint (default use the best checkpoint)')
    parser.add_argument('--no-extractor', action='store_true', help='do not use the intent extractor')
    parser.add_argument('--pooling', choices=['mean'], default='mean', help='how to aggregate hidden states')
    parser.add_argument('--no-list', action='store_true', help='do not include list names in input')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
