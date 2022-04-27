#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from os import makedirs
from os import path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import argparse
import json
import logging
import math
import os
import shutil

from torch import distributed as dist
from torch import multiprocessing as mp
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from transformers import AdamW
from transformers import BertModel
from transformers import BertTokenizer
from transformers import GPT2Model
from transformers import GPT2Tokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import PreTrainedTokenizer
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np

from src.dataset_readers import MultiTaskDataset
from src.models import SimpleCGenClfModel
from src.models import SimpleGenClfModel
from src.utils import dumps_config
from src.utils import freeze_module
from src.utils import open_helper
from src.utils import read_config
from src.utils import set_seed
from src.utils import unfreeze_module

TOKENIZER_CLASS = {
    'bert-base-cased': BertTokenizer,
    'bert-base-uncased': BertTokenizer,
    'bert-large-cased': BertTokenizer,
    'bert-large-uncased': BertTokenizer,
    'gpt2': GPT2Tokenizer,
    'roberta-base': RobertaTokenizer,
}
TRANSFORMER_CLASS = {
    'bert-base-cased#continuous': BertModel,
    'bert-base-cased#discrete': BertModel,
    'bert-base-uncased#continuous': BertModel,
    'bert-base-uncased#discrete': BertModel,
    'bert-large-cased#continuous': BertModel,
    'bert-large-cased#discrete': BertModel,
    'bert-large-uncased#continuous': BertModel,
    'bert-large-uncased#discrete': BertModel,
    'gpt2#continuous': GPT2Model,
    'gpt2#discrete': GPT2Model,
    'roberta-base#continuous': RobertaModel,
    'roberta-base#discrete': RobertaModel,
}


def get_logger(path_log: str, flush: Optional[bool] = False):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s/%(name)s[%(levelname)s]: %(message)s')
    logger.propagate = False
    if flush:
        with open(path_log, 'w') as f:
            f.write('')
    fh = logging.FileHandler(path_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def log(text:str, logger: Optional[logging.Logger] = None):
    try:
        logger.info(text)  # to log file
    except:
        pass
    tqdm.write(text)


def load_tokenizer(model_config: Dict[str, Any],
                   local_rank: Optional[int] = -1) -> PreTrainedTokenizer:
    if local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download a tokenizer
        dist.barrier()

    transformer_config = model_config.get('transformer', {})
    try:
        model_type = transformer_config.get('type')
        tokenizer_cls = TOKENIZER_CLASS[model_type]
    except KeyError:
        raise ValueError(f'model type {model_type} is invalid')
    tokenizer = tokenizer_cls.from_pretrained(model_type)
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'
    if tokenizer.bos_token is None:
        special_tokens['bos_token'] = '<s>' if tokenizer.cls_token is None \
            else tokenizer.cls_token
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '</s>'
    if tokenizer.sep_token is None:
        special_tokens['sep_token'] = '<sep>'
    if tokenizer.cls_token is None:
        special_tokens['cls_token'] = '<cls>'

    # Add special tokens
    tokenizer.add_special_tokens(special_tokens)

    if local_rank == 0:
        dist.barrier()

    return tokenizer


def create_model(model_config: Dict[str, Any],
                 tokenizer: PreTrainedTokenizer,
                 local_rank: Optional[int] = -1) -> torch.nn.Module:
    """Load a pretrained Transformer and create a model"""
    if local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download a model
        dist.barrier()

    transformer_config = model_config.get('transformer', {})
    try:
        model_type = transformer_config.get('type')
        primary_output = model_config.get('primary_output', 'discrete')
        _model_type = model_type + '#' + primary_output
        model_cls = TRANSFORMER_CLASS[_model_type]
    except KeyError:
        raise ValueError(f'model type {model_type} is invalid')

    kwargs = transformer_config.get('params', {})
    if 'model_path' in transformer_config:
        transformer = model_cls.from_pretrained(transformer_config['model_path'], **kwargs)
    else:
        transformer = model_cls.from_pretrained(model_type, **kwargs)
        # --> download a model if required
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
    if local_rank == 0:
        dist.barrier()

    return model


def get_optimizer(model: torch.nn.Module,
                  optimizer_config: Dict[str, Any],
                  logger: Optional[logging.Logger] = None) -> Optimizer:
    """Return an optimizer for a given model."""
    optimizer_type = optimizer_config.get('type', 'AdamW')
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0)
    log(f'Optimizer: {optimizer_config}', logger=logger)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    if optimizer_type == 'AdamW':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-6)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr,
                          betas=betas, eps=eps,
                          correct_bias=True)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(optimizer,
                  scheduler_config: Dict[str, Any],
                  num_training_steps: int,
                  logger: Optional[logging.Logger] = None) -> LambdaLR:
    """Return a scheduler for a given optimizer"""
    scheduler_type = scheduler_config.get('type', 'linear')
    warmup_steps_ratio = scheduler_config.get('warmup_steps_ratio', 0.002)
    num_warmup_steps = math.ceil(num_training_steps * warmup_steps_ratio)
    log(f'Scheduler: warmup={num_warmup_steps} {scheduler_config}', logger=logger)
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps)
    else:
        raise NotImplementedError
    return scheduler


def read_label_weights(weight_configs: List[Dict[str, str]],
                       indexers: Dict[str, int],
                       normalize: Optional[bool] = True,
                       logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    """Read label weights from an external JSON file.

    JSON format
    - label: label text
    - weight: label weight
    """
    if indexers is None or len(weight_configs) == 0:
        return {}
    weights  ={}
    for config in weight_configs:
        name = config['name']
        if name not in indexers:
            log(f'The aux task {name} does not have an indexer. Skip', logger=logger)
            continue
        indexer = indexers[name]
        weights[name] = torch.zeros(len(indexer))
        count = 0
        with open_helper(config['label_weight_file_path']) as f:
            for line in f:
                row = json.loads(line)
                if row['label'] not in indexer:
                    continue
                label_idx = indexer[row['label']]
                weights[name][label_idx] = row['weight']
                count += 1
        log(f'{name}: {count}/{len(indexer)} labels have weights', logger=logger)
        if normalize:  # normalize weights into [0, 1]
            weights[name] /= weights[name].max()
    return weights


def run_validation(model: torch.nn.Module,
                   dataloader: DataLoader,
                   tokenizer: PreTrainedTokenizer,
                   rank: Optional[int] = -1,
                   ac_loss_coef: Optional[float] = 1.0,
                   mtl_loss_coef: Optional[float] = 1.0,
                   label_weights: Optional[Dict[str, torch.Tensor]] = {},
                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run validation and return scores
    model: a model to be validated
    dataloader: a dataloader for reading a validation dataset
    rank: the rank of a device (-1 for single-process training)
    ac_loss_coef: loss weight for autocompletion
    mtl_loss_coef: loss weight for intent-focused tasks
    """
    model_device = next(model.parameters()).device
    iterator = enumerate(tqdm(dataloader, desc='Iteration', dynamic_ncols=True))
    model.eval()

    primary_output_type = model.primary_output_type if rank == -1 else model.module.primary_output_type

    scores= {
        f'vld/{category}': 0
        for category in ['loss', 'gen_loss', 'ppl',
                         'comet-xNeed_loss', 'comet-xIntent_loss',
                         'autosuggest_loss', 'framenet_loss',
                         'l2_dist', 'attnn', 'attnn_sep']
    }
    num_instances = defaultdict(int)  # for normalizing scores

    intent_embs, attnn, attnn_sep = [], [], []
    with torch.no_grad():
        for step, batch in iterator:
            num_instances['all'] += batch['input_ids'].size(0)
            for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
                if f'{key}_mask' in batch:
                    num_instances[key] += batch[f'{key}_mask'].sum().item()
            if model_device != 'cpu':
                for key, tensor in batch.items():
                    if not key.startswith('_'):
                        batch[key] = tensor.to(model_device)
            if rank == 0:
                outputs = model.module(batch, label_weights=label_weights)
                emb, _, attn, _ = model.module.get_intent_embs(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['input_mask'],
                    input_type_ids=batch.get('input_type_ids'),
                    return_attention_scores=True)
            else:
                outputs = model(batch, label_weights=label_weights)
                emb, _, attn, _ = model.get_intent_embs(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['input_mask'],
                    input_type_ids=batch.get('input_type_ids'),
                    return_attention_scores=True)
            attnn += attn[batch['input_ids'].ne(tokenizer.eos_token_id)&batch['input_mask']].cpu().numpy().tolist()
            attnn_sep += attn[batch['input_ids'].eq(tokenizer.eos_token_id)].cpu().numpy().tolist()
            intent_embs.append(emb.cpu())

            num_tokens = outputs['n'].unsqueeze(1)
            loss_gen = outputs['loss'].sum(dim=1, keepdim=True) / num_tokens

            scores['vld/loss'] += ac_loss_coef * loss_gen.sum().cpu().item()
            scores['vld/gen_loss'] += loss_gen.sum().cpu().item()
            if primary_output_type == 'discrete':
                scores['vld/ppl'] += loss_gen.exp().sum().cpu().item()

            for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
                loss = outputs.get(f'loss_{key}')
                if loss is None:
                    continue
                loss_sum = loss.sum().cpu().item()
                scores[f'vld/{key}_loss'] += loss_sum
                scores['vld/loss'] += mtl_loss_coef * loss_sum

            # Generation (sanity-check)
            if step == 0:
                input_type_ids = batch.get('input_type_ids')
                if input_type_ids is not None:
                    input_type_ids = input_type_ids[:3]
                if rank == -1:
                    ids = model.generate_greedy_top1(
                        input_ids=batch['input_ids'][:3],
                        attention_mask=batch['input_mask'][:3],
                        input_type_ids=input_type_ids,
                        max_length=16,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.bos_token_id)
                else:
                    ids = model.module.generate_greedy_top1(
                        input_ids=batch['input_ids'][:3],
                        attention_mask=batch['input_mask'][:3],
                        input_type_ids=input_type_ids,
                        max_length=16,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.bos_token_id)
                for in_ids, out_ids, gold_ids in zip(batch['input_ids'][:3], ids, batch['output_ids'][:3]):
                    result = tokenizer.decode(in_ids)
                    result += ' -> '
                    result += tokenizer.decode(out_ids)
                    result += ' // REFERENCE = '
                    result += tokenizer.decode(gold_ids)
                    log(result, logger=logger)
                if model_device != 'cpu':
                    torch.cuda.empty_cache()
    distances = []
    intent_embs = torch.cat(intent_embs, dim=0)
    for i in range(100):
        for j in range(100):
            distances.append(((intent_embs[0] - intent_embs[-1])**2).sum().sqrt().item())
    scores['vld/l2_dist'] = np.mean(distances)
    scores['vld/attnn'] = np.mean(attnn)
    scores['vld/attnn_sep'] = np.mean(attnn_sep)

    scores['vld/loss'] /= num_instances['all']
    scores['vld/gen_loss'] /= num_instances['all']
    scores['vld/ppl'] /= num_instances['all']
    for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
        if num_instances[key] > 0:
            scores[f'vld/{key}_loss'] /= num_instances[key]
    return scores


def train(rank: int, nprocs: int,
          config: Dict[str, Any],
          tokenizer: PreTrainedTokenizer) -> torch.nn.Module:

    if rank != -1:  # initialize distributed training env
        config['cuda'] = rank
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', init_method='env://',
                                rank=rank, world_size=nprocs)
    # Set up a logger (if None, no output to file)
    logger = get_logger(config['path_log']) if rank in [-1, 0] else None

    set_seed(seed=config['seed'], cuda=config['cuda'])
    if config['cuda'] >= 0:
        torch.cuda.set_device(config['cuda'])

    # Read data
    for dataset_config in config['dataset_reader']:
        split = dataset_config.get('split', 'train')
        assert 'filepath' in dataset_config
        if split == 'train':
            trn_dataset = MultiTaskDataset(tokenizer=tokenizer,
                                           local_rank=rank,
                                           is_validation=False,
                                           logger=logger,
                                           overwrite_cache=config['overwrite_cache'],
                                           **dataset_config)
        elif split == 'validation':
            if rank not in [-1, 0]:
                continue
            vld_dataset = MultiTaskDataset(tokenizer=tokenizer,
                                           local_rank=rank,
                                           logger=logger,
                                           overwrite_cache=config['overwrite_cache'],
                                           aux_label_indexers=trn_dataset.aux_label_indexers,
                                           **dataset_config)

    # Set up a model
    model_config = config.get('model', {})
    log(json.dumps(model_config, indent=2), logger=logger)
    model = create_model(model_config, tokenizer)
    if rank == -1 and config['cuda'] >= 0:
        model.to(config['cuda'])
    if rank != -1:
        model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)
    model_device = next(model.parameters()).device
    model.zero_grad()

    primary_output_type = model.primary_output_type if rank == -1 else model.module.primary_output_type

    # Save the tokenizer
    tokenizer.save_pretrained(path.join(config['save_dir'], 'tokenizer'))

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id

    def collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        batch_size = len(examples)
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
        if 'output_ids' in cols:
            seqs = [ex['output_ids'] for ex in examples]
            batch['output_ids'] = pad_sequence(
                seqs, padding_value=pad_token_id,
                batch_first=True)
            batch['output_mask'] = batch['output_ids'].ne(pad_token_id)

        for col in cols:
            if (not col.startswith('comet-xNeed') or not col.endswith('_ids')) \
               and (not col.startswith('comet-xIntent') or not col.endswith('_ids')) \
               and (not col.startswith('autosuggest') or not col.endswith('_ids')) \
               and (not col.startswith('framenet') or not col.endswith('_ids')):
                continue
            indices, seqs = zip(*[(i, ex[col]) for i, ex in enumerate(examples) if col in ex])
            if col.endswith('_target'):  # target vector
                batch[col] = torch.cat([seq.unsqueeze(0) for seq in seqs], dim=0)
            else:  # index vector
                batch[col] = pad_sequence(
                    seqs, padding_value=pad_token_id,
                    batch_first=True)
            mask = torch.zeros(batch_size, dtype=bool)
            mask[list(indices)] = True
            batch[col.rsplit('_', 1)[0] + '_mask'] = mask
        return batch

    trainer_config = config.get('trainer', {})
    batch_size = trainer_config.get('batch_size', 1)
    pretraining_num_epochs = trainer_config.get('pretraining_num_epochs', 0)
    num_epochs = trainer_config.get('num_epochs', 1)
    num_gradient_accumulation_steps = trainer_config.get('num_gradient_accumulation_steps', 1)
    if rank in [-1, 0]:
        log(f'batch size = {batch_size}', logger=logger)
        log(f'accum = {num_gradient_accumulation_steps}', logger=logger)
        log(f'actual batch size = {batch_size * num_gradient_accumulation_steps * nprocs}', logger=logger)
        log(f'pretraining epoch = {pretraining_num_epochs}', logger=logger)
        log(f'epoch = {num_epochs}', logger=logger)
    if num_gradient_accumulation_steps < 1:
         raise ValueError('"num_gradient_accumulation_steps" must be >= 1.')

    if rank == -1:
        trn_sampler = RandomSampler(trn_dataset)
        trn_dataloader = DataLoader(trn_dataset, sampler=trn_sampler,
                                    batch_size=batch_size,
                                    collate_fn=collate, drop_last=True,
                                    pin_memory=True)
    else:
        trn_sampler = DistributedSampler(trn_dataset,
                                         num_replicas=nprocs,
                                         rank=rank,
                                         shuffle=True,
                                         seed=config['seed'],
                                         drop_last=True)
        trn_dataloader = DataLoader(trn_dataset, sampler=trn_sampler,
                                    batch_size=batch_size,
                                    collate_fn=collate, drop_last=True,
                                    pin_memory=True)
    if rank in [-1, 0]:
        vld_sampler = SequentialSampler(vld_dataset)
        vld_dataloader = DataLoader(vld_dataset, sampler=vld_sampler,
                                    batch_size=batch_size,
                                    collate_fn=collate, drop_last=False,
                                    pin_memory=True)

    # Optimizer and scheduler
    optimizer_config = trainer_config.get('optimizer', {})
    max_grad_norm = optimizer_config.get('max_grad_norm')
    optimizer = get_optimizer(model, optimizer_config, logger=logger)
    scheduler_config = trainer_config.get('scheduler', {})
    if rank == -1:
        num_training_steps = len(trn_dataset) // batch_size * num_epochs
    else:
        num_training_steps = len(trn_dataset) // nprocs // batch_size * num_epochs
    num_training_steps //= num_gradient_accumulation_steps
    scheduler = get_scheduler(optimizer, scheduler_config, num_training_steps, logger=logger)

    if pretraining_num_epochs > 0:
        optimizer_config = trainer_config.get('pretraining_optimizer', {})
        pretraining_max_grad_norm = optimizer_config.get('max_grad_norm')
        pretraining_optimizer = get_optimizer(model, optimizer_config, logger=logger)
        scheduler_config = trainer_config.get('pretraining_scheduler', {})
        if rank == -1:
            num_training_steps = len(trn_dataset) // batch_size * pretraining_num_epochs
        else:
            num_training_steps = len(trn_dataset) // nprocs // batch_size * pretraining_num_epochs
        num_training_steps //= num_gradient_accumulation_steps
        pretraining_scheduler = get_scheduler(pretraining_optimizer, scheduler_config,
                                              num_training_steps, logger=logger)

    # AMP
    scaler = amp.GradScaler()

    # For loss calculation
    ac_loss_coef = trainer_config.get('ac_coef', 1.0)
    mtl_loss_coef = trainer_config.get('mtl_coef', 1.0)
    mtl_weighting_method = trainer_config.get('mtl_weighting_method')

    label_weights = None
    if isinstance(trainer_config.get('label_weight'), list):
        label_weights = read_label_weights(trainer_config['label_weight'],
                                           trn_dataset.aux_label_indexers,
                                           normalize=True,
                                           logger=logger)
        if model_device != 'cpu':
            for label, weights in label_weights.items():
                label_weights[label] = weights.to(model_device)

    log(f'ac_coef: {ac_loss_coef}', logger=logger)
    log(f'mtl_coef: {mtl_loss_coef}', logger=logger)
    log(f'mtl_w_method: {mtl_weighting_method}', logger=logger)
    log(f'label_weights: {trainer_config.get("label_weight")}', logger=logger)

    # Logging
    global_step, update_step = 0, 0
    if os.path.isdir(config['log_dir']):  # remove the directory if exists
        shutil.rmtree(config['log_dir'])
    if rank == -1:
        tb_writer = SummaryWriter(log_dir=config['log_dir'])
    else:
        tb_writer = SummaryWriter(log_dir=path.join(config['log_dir'], f'p{rank}'))
    patience = trainer_config.get('patience', 1)
    best_vld_loss, prev_vld_loss, patience_counter = None, None, 0
    performance = []
    if rank in [-1, 0]:
        epoch_iterator = trange(1, pretraining_num_epochs+num_epochs+1,
                                desc='Epoch', dynamic_ncols=True)
    else:
        epoch_iterator = range(1, pretraining_num_epochs+num_epochs+1)

    if pretraining_num_epochs > 0:  # do not update transformer for `pretraining_num_epochs`
        freeze_module(model.transformer if rank == -1 else model.module.transformer)
    for epoch in epoch_iterator:
        perf = {
            f'{split}/{category}': 0
            for split in ['trn', 'vld']
            for category in ['loss', 'gen_loss', 'ppl',
                             'comet-xNeed_loss', 'comet-xIntent_loss',
                             'autosuggest_loss', 'framenet_loss',
                             'l2_dist', 'attnn', 'attnn_sep']
        }
        # Training
        if rank != -1:
            trn_sampler.set_epoch(epoch)
        if rank in [-1, 0]:
            iterator = tqdm(trn_dataloader, desc='Iteration', dynamic_ncols=True)
        else:
            iterator = trn_dataloader
        model.train()
        num_instances = defaultdict(int)
        if epoch == pretraining_num_epochs+1:
            # Unfreeze the transformer module
            unfreeze_module(model.transformer if rank == -1 else model.module.transformer)
            tqdm.write('Unfreeze transformer')
        is_pretraining_phase = (epoch <= pretraining_num_epochs)
        # DEBUG
        tqdm.write(f'{epoch} is_pretraining_phase={is_pretraining_phase}')
        for step, batch in enumerate(iterator):
            global_step += 1
            num_instances['all'] += batch['input_ids'].size(0)
            for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
                if f'{key}_mask' in batch:
                    num_instances[key] += batch[f'{key}_mask'].sum().item()
            if model_device != 'cpu':
                for key, tensor in batch.items():
                    if key.startswith('_'):
                        continue
                    batch[key] = tensor.to(model_device)

            # with amp.autocast():
            outputs = model(batch, label_weights=label_weights)

            if mtl_weighting_method == 'muppet' and ac_loss_coef > 0:
                if rank == -1:
                    ac_loss_coef = 1 / math.log(model.lm_head.out_features)
                else:
                    ac_loss_coef = 1 / math.log(model.module.lm_head.out_features)
            num_tokens = outputs['n'].unsqueeze(1)
            loss_gen = outputs['loss'].sum(dim=1, keepdim=True) / num_tokens
            total_loss = ac_loss_coef * loss_gen.mean()

            perf['trn/loss'] += ac_loss_coef * loss_gen.detach().sum().cpu().item()
            perf['trn/gen_loss'] += loss_gen.detach().sum().cpu().item()
            if primary_output_type == 'discrete':
                perf['trn/ppl'] += loss_gen.detach().exp().sum().cpu().item()

            for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
                loss = outputs.get(f'loss_{key}')
                if loss is None:
                    continue
                if mtl_weighting_method == 'muppet':
                    if rank == -1:
                        mtl_loss_coef = 1 / math.log(model.aux_num_labels[key])
                    else:
                        mtl_loss_coef = 1 / math.log(model.module.aux_num_labels[key])
                total_loss = total_loss + mtl_loss_coef * loss.mean()
                loss_sum = loss.detach().sum().cpu().item()
                perf[f'trn/{key}_loss'] += loss_sum
                perf['trn/loss'] += mtl_loss_coef * loss_sum

            (total_loss.mean() / num_gradient_accumulation_steps).backward()
            # scaler.scale(total_loss.mean() / num_gradient_accumulation_steps).backward()

            if global_step % num_gradient_accumulation_steps != 0:
                continue

            if max_grad_norm is not None:
                # scaler.unscale_(optimizer)
                if is_pretraining_phase:
                    clip_grad_norm_(model.parameters(), pretraining_max_grad_norm)
                else:
                    clip_grad_norm_(model.parameters(), max_grad_norm)

            update_step += 1
            # scaler.step(optimizer)
            # if update_step == 1:
            #     # at step=1, grad=0 and scaler does not call optimzier.step()
            #     for param in model.parameters():
            #         param.grad = None
            #     # Run optimizer.step() to suppress an error from LR scheduler
            #     if is_pretraining_phase:
            #         pretraining_optimizer.step()
            #     else:
            #         optimizer.step()
            if is_pretraining_phase:
                pretraining_optimizer.step()
                pretraining_scheduler.step()
            else:
                optimizer.step()
                scheduler.step()
            # scaler.update()

            # Memory-efficient zero grad
            for param in model.parameters():
                param.grad = None

            if global_step % 100 == 0:
                for category in ['loss', 'ppl']:
                    trn = perf[f'trn/{category}'] / num_instances['all']
                    tb_writer.add_scalar(f'{category}/trn', trn, update_step)
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], update_step)
                if model_device != 'cpu':
                    torch.cuda.empty_cache()

        perf['trn/loss'] /= num_instances['all']
        perf['trn/gen_loss'] /= num_instances['all']
        perf['trn/ppl'] /= num_instances['all']
        for key in ['comet-xNeed', 'comet-xIntent', 'autosuggest', 'framenet']:
            if num_instances[key] > 0:
                perf[f'trn/{key}_loss'] /= num_instances[key]

        # Validation
        if rank in [-1, 0]:
            scores = run_validation(model, vld_dataloader, tokenizer,
                                    rank=rank,
                                    ac_loss_coef=ac_loss_coef,
                                    mtl_loss_coef=mtl_loss_coef,
                                    label_weights=label_weights,
                                    logger=logger)
            perf.update(scores)

            if not is_pretraining_phase:
                if best_vld_loss is None or best_vld_loss > perf['vld/loss']:
                    save_path = path.join(config['save_dir'], 'model.best.pth')
                    if best_vld_loss is not None:
                        log(f'Best validation loss'
                            f' {perf["vld/loss"]:.6f}'
                            f' <= {best_vld_loss:.6f}'
                            f' Saving the checkpoint to {save_path}',
                            logger=logger)
                    best_vld_loss = perf['vld/loss']
                    patience_counter = 0
                    if rank == -1:
                        torch.save(model.state_dict(), save_path)
                    else:
                        torch.save(model.module.state_dict(), save_path)
                elif prev_vld_loss < perf['vld/loss']:
                    patience_counter += 1
                    if patience_counter == patience:
                        log(f'EXIT: patience_counter == {patience}', logger=logger)
                        break
                else:
                    patience_counter = 0
            prev_vld_loss = perf['vld/loss']

        if rank != -1:
            dist.barrier()  # sync

        # Logging
        buff = f'[{epoch}]: train/validation '
        if rank != -1:
            buff = f'({rank}) ' + buff
        for category in ['loss', 'gen_loss', 'ppl',
                         'comet-xNeed_loss', 'comet-xIntent_loss',
                         'autosuggest_loss', 'framenet_loss']:
            trn, vld = perf[f'trn/{category}'], perf[f'vld/{category}']
            buff += f'{category}={trn:.6f}/{vld:.6f} '
            tb_writer.add_scalar(f'{category}/trn', trn, update_step)
            tb_writer.add_scalar(f'{category}/vld', vld, update_step)
        buff += f'l2_dist= {perf["vld/l2_dist"]:.6f} '
        tb_writer.add_scalar(f'l2_dist/vld', perf["vld/l2_dist"], update_step)
        buff += f'attnn= {perf["vld/attnn"]:.6f} '
        tb_writer.add_scalar(f'attnn/vld', perf["vld/attnn"], update_step)
        buff += f'attnn_sep= {perf["vld/attnn_sep"]:.6f} '
        tb_writer.add_scalar(f'attnn_sep/vld', perf["vld/attnn_sep"], update_step)
        try:
            buff += f'lr={pretraining_optimizer.param_groups[0]["lr"]}/{optimizer.param_groups[0]["lr"]} '
        except UnboundLocalError:
            buff += f'lr={optimizer.param_groups[0]["lr"]} '
        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], update_step)
        buff += f'patience={patience_counter}'
        log(buff.strip(), logger=logger)

        perf['epoch'] = epoch
        perf['update_step'] = update_step
        performance.append(perf)

    # Save the scores to score.json
    path_score = path.join(config['save_dir'], 'scores.json' if rank == -1 else f'scores{rank}.json')
    with open(path_score, 'w') as f:
        json.dump(performance, f, indent=4)

    # Clean up
    tb_writer.close()
    # Do not save the last model
    # if rank in [0, -1]:
    #     save_path = path.join(config['save_dir'], 'model.pth')
    #     if rank == 0:
    #         torch.save(model.module.state_dict(), save_path)
    #     else:
    #         torch.save(model.state_dict(), save_path)

    if rank != -1:
        dist.destroy_process_group()

    return model


def main(args):
    # Create an output directory
    makedirs(args.save_dir, exist_ok=True)

    # Set up a logger & log file
    path_log = path.join(args.save_dir, 'train.log')
    logger = get_logger(path_log, flush=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = read_config(args.path_config)

    # Load a pretrained tokenizer
    model_config = config.get('model', {})
    tokenizer = load_tokenizer(model_config)

    # Save the config
    with open(path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    config_str = dumps_config(config)
    config_str = path.basename(args.save_dir) + '-' + config_str
    config_str = config_str.replace('/', '-')
    config['log_dir'] = path.join(args.log_dir, config_str)
    config['save_dir'] = args.save_dir
    config['cuda'] = args.cuda
    config['seed'] = args.seed
    config['path_log'] = path_log
    config['overwrite_cache'] = args.overwrite_cache

    nprocs = torch.cuda.device_count()
    if nprocs <= 1:
        raise ValueError(f'--parallel is given but # of GPUs = {nprocs}')
    log(f'{nprocs} GPUs are available', logger=logger)
    for handler in logger.handlers:  # close the log file
        handler.close()
        logger.removeHandler(handler)
    if args.parallel:
        mp.spawn(train,
                 args=(nprocs, config, tokenizer),
                 nprocs=nprocs,
                 join=True)
    else:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        # rank = -1 (single process)
        # nprocs = 1 (# of processes)
        train(-1, 1, config, tokenizer)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_config', help='path to a config file')
    parser.add_argument('-s', '--save-dir', type=str, required=True, help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log-dir', type=str, default='logs', help='log directory')
    parser.add_argument('--overwrite-cache', action='store_true', help='overwrite data cache')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('--parallel', action='store_true', help='Train a model on multiple GPUs')
    args = parser.parse_args()
    main(args)
