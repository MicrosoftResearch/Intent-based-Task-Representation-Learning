#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
from typing import List
from typing import Optional
from typing import Tuple
import argparse

from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import GPT2Model
from transformers import GPT2Tokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch import no_grad
from torch import LongTensor
from torch import Tensor
from torch import cat

from baseline_encoders.utils import init_logger
from baseline_encoders.dataset_readers import read_CoTL
from baseline_encoders.utils import batchfy
from baseline_encoders.utils import open_helper

verbose = False

TRANSFORMER_CLASS = {
    'bert-base-cased': (BertModel, BertTokenizer),
    'bert-base-uncased': (BertModel, BertTokenizer),
    'bert-large-cased': (BertModel, BertTokenizer),
    'bert-large-uncased': (BertModel, BertTokenizer),
    'roberta-base': (RobertaModel, RobertaTokenizer),
    'gpt2': (GPT2Model, GPT2Tokenizer),
}
def load_transformer(model_type: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    try:
        model_cls, tokenizer_cls = TRANSFORMER_CLASS[model_type]
        model = model_cls.from_pretrained(model_type)
        tokenizer = tokenizer_cls.from_pretrained(model_type)
    except KeyError:
        assert path.isdir(model_type)
        model = AutoModel.from_pretrained(model_type)
        if 'roberta' in model_type:
            tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)

    if isinstance(model, GPT2Model):
        # Add PAD token to the vocabulary
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer


def encode(model: PreTrainedModel, input_ids: Tensor, input_mask: Tensor) -> Tensor:
    """Encode input tokens and return the last hidden states."""
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    input_mask = input_mask.to(model_device)
    if model.config.is_encoder_decoder:
        encoder = model.get_encoder()
        out = encoder(input_ids=input_ids, attention_mask=input_mask)
        return out['last_hidden_state'].cpu()
    out = model(input_ids=input_ids, attention_mask=input_mask)
    return out['last_hidden_state']


def aggregate_embs(embs: Tensor, mask: Tensor, how: str) -> Tensor:
    if how == 'cls': # take the CLS embeddings:
        return embs[:, 0]
    if how == 'mean':  # mean average pooling
        num_tokens = mask.sum(dim=1, keepdim=True)
        embs[~mask] = 0  # set the embeddings of pad tokens to zero
        return embs.sum(dim=1) / num_tokens
    if how == 'sum':  # sum pooling
        return embs.sum(dim=1)
    if how == 'max':  # max pooling
        num_tokens = mask.sum(dim=1, keepdim=True)
        embs[~mask] = embs.min()  # set the embeddings of pad tokens to the min value
        return embs.max(dim=1).values

    raise NotImplementedError

def main(args):
    global verbose
    verbose = args.verbose

    # Load a Transformer model
    if verbose:
        logger.info(f'Model type: {args.model_type}')
    model, tokenizer = load_transformer(args.model_type)
    if args.cuda >= 0:
        model.to(args.cuda)

    # Read a dataset
    if verbose:
        logger.info(f'Data type: {args.data_type}')
    if args.data_type == 'CoTL':
        reader = read_CoTL(args.path_input)
    else:
        raise NotImplementedError

    # Encode text
    if verbose:
        logger.info(f'Encode text')
        logger.info(f'How to concatenate task and list names: {args.concat}')
        logger.info(f'How to aggregate embeddings: {args.pooling}')
    reader_ = batchfy(reader, batchsize=args.batchsize)
    bos_token_id = tokenizer.bos_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    eos_token_id = tokenizer.eos_token_id
    encoded = []

    def add_special_token_ids(ids: List[int],
                              ids2: Optional[List[int]] = None):
        if args.model_type == 'gpt2':
            # GPT2 does not use BOS nor EOS
            ret = ids
            if ids2 is not None:
                ret += ids2
            return ret
        ret = [] if bos_token_id is None else [bos_token_id]
        if bos_token_id is None and cls_token_id is not None:
            ret.append(cls_token_id)
        ret += ids
        if ids2 is not None:
            if sep_token_id is not None:
                ret.append(sep_token_id)
            ret += ids2
        if eos_token_id is not None:
            ret.append(eos_token_id)
        elif sep_token_id is not None:
            ret.append(sep_token_id)
        return ret

    with no_grad():
        for batch in tqdm(reader_):
            tasks1, lists1, tasks2, lists2, _ = zip(*batch)
            encoded_ = []
            for tasks, lists in [(tasks1, lists1), (tasks2, lists2)]:
                # Tokenization
                input_tokens, input_ids = None, None
                tasks_ids = [
                    tokenizer(text.replace('#OOV#', tokenizer.unk_token).split(),
                              is_split_into_words=True, add_special_tokens=False)['input_ids']
                    for text in tasks]
                if lists[0] is None:  # No list
                    input_ids = [add_special_token_ids(ids) for ids in tasks_ids]
                else:
                    lists_ids = [
                        tokenizer(text.replace('#OOV#', tokenizer.unk_token).split(),
                                  is_split_into_words=True, add_special_tokens=False)['input_ids']
                        for text in lists]
                    if args.concat == 'input':
                        input_ids = [add_special_token_ids(tasks_, lists_)
                                     for tasks_, lists_ in zip(tasks_ids, lists_ids)]

                # Encode
                if input_ids is not None:
                    input_ids = pad_sequence(
                        [LongTensor(ids) for ids in input_ids],
                        padding_value=tokenizer.pad_token_id,
                        batch_first=True)
                    input_mask = input_ids.ne(tokenizer.pad_token_id)
                    if args.cuda >= 0:
                        input_ids = input_ids.to(args.cuda)
                        input_mask = input_mask.to(args.cuda)
                    embs = encode(model, input_ids, input_mask)
                    pooled_embs = aggregate_embs(embs, input_mask, how=args.pooling)
                else:
                    pooled_embs = []
                    for ids in [tasks_ids, lists_ids]:
                        input_ids = pad_sequence(
                            [LongTensor(ids_) for ids_ in ids],
                            padding_value=tokenizer.pad_token_id,
                            batch_first=True)
                        input_mask = input_ids.ne(tokenizer.pad_token_id)
                        if args.cuda >= 0:
                            input_ids = input_ids.to(args.cuda)
                            input_mask = input_mask.to(args.cuda)
                        embs = encode(model, input_ids, input_mask)
                        pooled_embs.append(aggregate_embs(embs, input_mask, how=args.pooling))
                    pooled_embs = cat(pooled_embs, dim=1)

                encoded_ += [tasks, lists, pooled_embs.cpu().numpy().tolist()]
            encoded += list(zip(*encoded_))

    # Output
    if verbose:
        logger.info(f'Write {len(encoded)} embeddings to {args.path_output}')
    with open_helper(args.path_output, 'w') as f:
        dim = len(encoded[0][-1]) * 2
        f.write(f'{len(encoded)} {dim}\n')
        for task1, list1, emb1, task2, list2, emb2 in encoded:
            emb1 = ' '.join(map(str, emb1))
            emb2 = ' '.join(map(str, emb2))
            if list1 is None and list2 is None:
                f.write(f'{task1.replace(" ", "_")} {task2.replace(" ", "_")} {emb1} {emb2}\n')
            else:
                f.write(f'{task1.replace(" ", "_")}@@@{list2.replace(" ", "_")}@@@{task1.replace(" ", "_")}@@@{list2.replace(" ", "_")} {emb1} {emb2}\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('BaselineTransformers')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--data-type', choices=['CoTL'], required=True, help='Dataset type')
    parser.add_argument('--dummy-list', help='dummy to-do list name for UIT')
    parser.add_argument('--model-type', required=True, help='Transformer model type')
    parser.add_argument('--concat', choices=['input', 'output'], default='input', help='how to concatenate task and list names. output will double the dimension size')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'max', 'sum'], default='mean', help='how to aggregate token embeddings')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to an output file')
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
