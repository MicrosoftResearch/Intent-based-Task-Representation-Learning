#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
from os import path
import argparse

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig
from transformers import BartForConditionalGeneration
from transformers import BartModel
from transformers import BartTokenizer
from transformers import BertModel
from transformers import BertTokenizer
from transformers import GPT2Model
from transformers import GPT2Tokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizer
import torch

from src.dataset_readers import CoTLDataset
from src.models import BartCGenerationClfModel
from src.models import BartGenerationClfModel
from src.models import SimpleCGenClfModel
from src.models import SimpleGenClfModel
from src.utils import init_logger
from src.utils import open_helper
from src.utils import read_config

TOKENIZER_CLASS = {
    'bert-base-cased': BertTokenizer,
    'bert-base-uncased': BertTokenizer,
    'bert-large-cased': BertTokenizer,
    'bert-large-uncased': BertTokenizer,
    'facebook/bart-base': BartTokenizer,
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
    'facebook/bart-base#continuous': BartModel,
    'facebook/bart-base#discrete': BartForConditionalGeneration,
    'gpt2#continuous': GPT2Model,
    'gpt2#discrete': GPT2Model,
    'roberta-base#continuous': RobertaModel,
    'roberta-base#discrete': RobertaModel,
}

DATASET_CLASS = {
    'cotl': CoTLDataset,
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
    wrapper_model_type = model_config.get('type', 'encoder_decoder')
    if wrapper_model_type == 'encoder_decoder':
        if primary_output == 'discrete':
            model = BartGenerationClfModel(transformer, tokenizer, **model_config)
        else:
            model = BartCGenerationClfModel(transformer, tokenizer, **model_config)
    elif wrapper_model_type == 'simple_encoder':
        if primary_output == 'discrete':
            model = SimpleGenClfModel(transformer, tokenizer, **model_config)
        else:
            model = SimpleCGenClfModel(transformer, tokenizer, **model_config)
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
    dataset = dataset_cls(args.path_input, tokenizer, use_cache=False, **dataset_config)

    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    def collate(examples):
        batch = {}
        cols = set(chain.from_iterable([ex.keys() for ex in examples]))
        for i in [1, 2]:
            seqs = [ex[f'input_ids{i}'] for ex in examples]
            if f'input_type_ids{i}' in cols:
                type_seqs = [ex[f'input_type_ids{i}'] for ex in examples]
                batch[f'input_type_ids{i}'] = pad_sequence(type_seqs,
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
                batch[f'input_type_ids{i}'] = pad_sequence(token_types,
                                                           padding_value=0,
                                                           batch_first=True)
            batch[f'input_ids{i}'] = pad_sequence(seqs, padding_value=pad_token_id,
                                                  batch_first=True)
            batch[f'input_mask{i}'] = batch[f'input_ids{i}'].ne(pad_token_id)
            batch[f'_task{i}'] = [ex[f'task{i}'] for ex in examples]
            batch[f'_list{i}'] = [ex[f'list{i}'] for ex in examples]
            batch[f'_task.full{i}'] = [ex.get(f'task.full{i}', '') for ex in examples]

        return batch

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=args.batch_size,
                            collate_fn=collate, drop_last=False)

    if args.cuda >= 0:
        model = model.to(args.cuda)

    buff = {col: [] for col in ['task1', 'list1', 'emb1',
                                'task2', 'list2', 'emb2']}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.cuda >= 0:
                for key, val in batch.items():
                    if key.startswith('_'):
                        continue
                    batch[key] = val.to(args.cuda)
            for i in [1, 2]:
                if args.no_extractor:
                    try:
                        encoder = model.transformer.get_encoder()
                        out = encoder(
                            input_ids=batch[f'input_ids{i}'],
                            attention_mask=batch[f'input_mask{i}'])
                    except AttributeError:
                        out = model.transformer(
                            input_ids=batch[f'input_ids{i}'],
                            attention_mask=batch[f'input_mask{i}'])
                    h = out['last_hidden_state']
                    # batch['input_mask'] = (batch['input_mask'] & (batch['input_type_ids'] != 2))
                    h[~batch[f'input_mask{i}']] = 0
                    if args.pooling == 'mean':
                        intent_embs = h.sum(dim=1) / batch[f'input_mask{i}'].sum(dim=1, keepdim=True)
                    else:
                        raise NotImplementedError
                else:
                    intent_embs = model.get_intent_embs(
                        input_ids=batch[f'input_ids{i}'],
                        attention_mask=batch[f'input_mask{i}'],
                        input_type_ids=batch.get(f'input_type_ids{i}'))
                if intent_embs.device != 'cpu':
                    intent_embs = intent_embs.cpu()

                for idx, (taskname, listname, taskname_full) in enumerate(
                        zip(batch[f'_task{i}'], batch[f'_list{i}'], batch[f'_task.full{i}'])):
                    buff[f'task{i}'].append(taskname.replace(' ', '_'))
                    buff[f'list{i}'].append(listname.replace(' ', '_'))
                    buff[f'emb{i}'].append(' '.join(map(str, intent_embs[idx].numpy().tolist())))

    logger.info(f'Write to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        f.write(f'{len(buff["task1"])} {len(buff["emb1"][0].split()) * 2}\n')
        for idx in range(len(buff['task1'])):
            taskname1 = buff['task1'][idx]
            listname1 = buff['list1'][idx]
            emb1 = buff['emb1'][idx]
            taskname2 = buff['task2'][idx]
            listname2 = buff['list2'][idx]
            emb2 = buff['emb2'][idx]
            f.write(f'{taskname1}@@@{listname1}###{taskname2}@@@{listname2} {emb1} {emb2}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('Generation')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to a input file')
    parser.add_argument('--data-type', choices=['cotl'], default='cotl', help='Dataset type')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to output file')
    parser.add_argument('-m', '--model-dir', type=str, required=True, help='model directory')
    parser.add_argument('--use-final', action='store_true', default=False, help='use the final checkpoint (default use the best checkpoint)')
    parser.add_argument('--no-extractor', action='store_true', help='do not use the intent extractor')
    parser.add_argument('--pooling', choices=['mean'], default='mean', help='how to aggregate hidden states')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
