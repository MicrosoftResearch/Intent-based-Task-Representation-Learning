# -*- coding: utf-8 -*-

from os import path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import math
import sys

from torch import nn
from transformers import BertModel
from transformers import GPT2Model
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
import torch
import numpy as np

from ..modules import FeedForward
from ..modules import GILE
from ..modules import LabelSmoothedCELoss
from ..modules import MultiHeadAttentionWithFixedQuery


def get_activation_func(name: str) -> nn.Module:
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == 'none':  # no activation
        return nn.Identity()
    raise ValueError(f'activation "{name}" is invalid.')


class SimpleGenClfBaseModel(nn.Module):
    def __init__(self,
                 transformer_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 intent_extractor: Optional[Dict[str, Any]] = {},
                 decoder: Optional[Dict[str, Any]] = {},
                 auxiliary: Optional[List[Dict[str, Any]]] = [],
                 input_type_embeddings: Optional[bool] = False,
                 **kwargs) -> None:
        super().__init__()
        if transformer_model.config.is_encoder_decoder:
            raise ValueError('Encoder-decoder Transformer models cannot be used')

        self.transformer = transformer_model
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id

        d_model = self.transformer.config.hidden_size

        self.input_type_embeddings = input_type_embeddings
        if self.input_type_embeddings:
            # Resize the type embedding layer
            self.transformer.config.type_vocab_size = 3
            self.transformer.embeddings.token_type_embeddings = nn.Embedding(3, d_model)
            self.transformer.embeddings.token_type_embeddings.weight.data.zero_()

        # Intent extractor
        ext_conf = intent_extractor
        ## Input type embeddings (0: special symbol, 1: task, 2: list)
        self.type_emb_dim = ext_conf.get('type_embedding_dim', 0)
        if self.type_emb_dim > 0:
            self.type_embs = nn.Embedding(3, self.type_emb_dim, padding_idx=0)
        ## Attention module
        attn_conf = ext_conf.get('attention', {})
        self.num_attention_heads = attn_conf.get('num_attention_heads', 1)
        self.attention_dropout = attn_conf.get('dropout', 0)
        self.attention_type = attn_conf.get('type', 'additive')
        self.attention_input_size = d_model
        self.proj_h, self.proj_t = None, None
        if attn_conf.get('proj_h', False) and self.attention_type != 'bilinear':
            self.proj_h = nn.Linear(d_model, d_model)
        if attn_conf.get('proj_t', False):
            self.proj_t = nn.Linear(self.type_emb_dim, self.type_emb_dim)
        if self.attention_type == 'additive':
            self.attention_input_size += self.type_emb_dim
            self.W = nn.Linear(self.attention_input_size, 1, bias=False)   # attention weights
        elif self.attention_type == 'sum':
            assert self.type_emb_dim == 0 or self.type_emb_dim == d_model
            self.W = nn.Linear(self.attention_input_size, 1, bias=False)   # attention weights
        elif self.attention_type == 'bilinear':
            bias = attn_conf.get('bias', False)
            self.proj_h = nn.Linear(d_model, self.type_emb_dim)
            self.W = nn.Linear(self.type_emb_dim, 1, bias=False)   # attention weights
        elif self.attention_type == 'none':
            pass # No attention
        else:
            raise NotImplementedError
        self.attention_activation = get_activation_func(attn_conf.get('activation', 'none'))
        self.attention = None
        if self.attention_type not in {'none', 'bilinear'} and self.num_attention_heads > 0:
            self.attention = MultiHeadAttentionWithFixedQuery(
                input_size=self.attention_input_size,
                num_attention_heads=self.num_attention_heads,
                dropout=self.attention_dropout,
            )

        ## Extractor module
        num_layers = ext_conf.get('num_layers', 0)
        hidden_dims = ext_conf.get('hidden_dims', 1)
        activations = ext_conf.get('activations', 'none')
        dropout = ext_conf.get('dropout', 0)
        input_activation = ext_conf.get('input_activation', 'none')
        input_dropout = ext_conf.get('input_dropout', 0)
        bias = ext_conf.get('bias', True)
        initialize_method = ext_conf.get('initialization', 'normal')
        if num_layers > 0:
            self.extractor = FeedForward(d_model,
                                         num_layers, hidden_dims,
                                         activations, dropout, bias=bias,
                                         input_activation=input_activation,
                                         input_dropout=input_dropout)
            intent_emb_dim = self.extractor.get_output_dim()
        else:
            self.extractor = nn.Identity()
            intent_emb_dim = d_model

        # Intent-focused auxiliary tasks
        aux_modules, aux_label_embedders = {}, {}
        self.aux_output_type, self.aux_num_labels = {}, {}
        for aux_conf in auxiliary:
            name = aux_conf['name']
            module_type = aux_conf.get('type', 'ffn')
            self.aux_output_type[name] = aux_conf.get('output_type', 'cbol')
            input_dim = intent_emb_dim
            if module_type in {'ffn', 'gru'}:  # required args for FFN
                num_layers = aux_conf['num_layers']
                hidden_dims = aux_conf['hidden_dims']
            activations = aux_conf.get('activations', 'none')
            dropout = aux_conf.get('dropout', 0)
            label_emb_dim = d_model  # size of token embedding
            if name == 'framenet':
                if aux_conf.get('vocab_size', 0) > 0:
                    vocab_size = aux_conf['vocab_size'] + 2  # PAD + UNK
                    label_emb_dim = aux_conf.get('label_embedding_dim', d_model)
                    label_embedder = nn.Embedding(vocab_size, label_emb_dim)
                    if isinstance(aux_conf.get('label_embedding_path'), str):
                        self.initialize_embedder(label_embedder,
                                                 aux_conf['label_embedding_path'])
                    if aux_conf.get('freeze_label_embedding', False):
                        label_embedder.weight.requires_grad = False
                    aux_label_embedders[name] = label_embedder
            if module_type == 'gile':
                joint_dim = aux_conf.get('joint_dim', 100)
                assert isinstance(activations, str)
                aux_modules[name] = GILE(input_dim=intent_emb_dim,
                                         label_dim=label_emb_dim,
                                         joint_dim=joint_dim,
                                         activation=activations,
                                         dropout=dropout,
                                         bias=True)
            elif module_type == 'ffn':
                aux_modules[name] = FeedForward(input_dim, num_layers, hidden_dims,
                                                activations, dropout)
                assert aux_modules[name].get_output_dim() == label_emb_dim
            elif module_type == 'gru':
                bias = aux_conf.get('bias', True)
                pre_decoder = nn.ModuleList(
                    [FeedForward(input_dim, 1, input_dim,
                                 activations, dropout, bias=bias)
                     for _ in range(num_layers)])
                decoder = nn.GRU(input_dim, hidden_dims, num_layers, bias,
                                 batch_first=True, dropout=dropout,
                                 bidirectional=False)
                lm_head = nn.Linear(
                    hidden_dims[-1] if isinstance(hidden_dims, list) else hidden_dims,
                    self.transformer.get_input_embeddings().num_embeddings,
                    bias=False)
                aux_modules[f'{name}_pre_decoder'] = pre_decoder
                aux_modules[name] = decoder
                aux_modules[f'{name}_lm_head'] = lm_head
            else:
                raise NotImplementedError
            if self.aux_output_type[name] == 'cbol':
                self.aux_num_labels[name] = aux_modules[name].get_output_dim()
            elif self.aux_output_type[name] == 'multilabel':
                self.aux_num_labels[name] = vocab_size - 2
            elif self.aux_output_type[name] == 'generation':
                self.aux_num_labels[name] = self.transformer.get_input_embeddings().num_embeddings
            else:
                raise NotImplementedError

        self.aux_modules = nn.ModuleDict(aux_modules)
        self.aux_label_embedders = nn.ModuleDict(aux_label_embedders)
        self.initialize_intent_modules(how=initialize_method,
                                       std=ext_conf.get('initialization_std', 0.02))

        # Loss functions for auxiliary tasks
        self.loss_mse_fn = nn.MSELoss(reduction='none')
        self.loss_bcel_fn = nn.BCEWithLogitsLoss(reduction='none')

    def initialize_embedder(self,
                            embedder: nn.Embedding,
                            filepath: Optional[str],
                            add_unk: Optional[bool] = True,
                            add_pad: Optional[bool] = True) -> None:
        if not path.isfile(filepath):
            sys.stderr.write(f'{filepath} does not exist. The label embeddings are not initialized with an embedding file. (You can ignore this warning if you\'re loading a pre-trained model file.\n')
            return
        with open(filepath) as f:
            # Input format: word2vec format
            next(f)  # skip a header
            embs = []
            for line in f:
                tok, emb = line.split(' ', 1)
                embs.append([float(v) for v in emb.split()])
        embs = torch.FloatTensor(np.array(embs))
        avg = embs.mean(dim=0)
        i = 0
        if add_unk:
            embedder.weight.data[i] = avg
            i += 1
        if add_pad:
            embedder.weight.data[i] = avg
            i += 1
        embedder.weight.data[i:] = embs
        # These are in-place operations

    def initialize_intent_modules(self, how: Optional[str] = 'normal',
                                  std: Optional[float] = 0.02) -> None:
        if self.type_emb_dim > 0:
            self.type_embs.weight.data.normal_(mean=0.0, std=std)
        if self.attention is not None:
            if how == 'normal':
                self.attention.initialize_normal(mean=0, std=std)
            else:
                raise NotImplementedError
        if self.attention_type != 'none':
            if how == 'normal':
                self.W.weight.data.normal_(mean=0.0, std=std)
                if isinstance(self.extractor, FeedForward):
                    self.extractor.initialize_weights(std=std)
            elif how == 'average':
                self.W.weight.data.fill_(0)  # start with average pooling
                if isinstance(self.extractor, FeedForward):
                    self.extractor.initialize_weights(std=std, identity=True)
            else:
                raise NotImplementedError
        if isinstance(self.proj_h, nn.Module):
            self.proj_h.weight.data.normal_(mean=0.0, std=std)
            if self.proj_h.bias is not None:
                self.proj_h.bias.data.zero_()
        if isinstance(self.proj_t, nn.Module):
            self.proj_t.weight.data.normal_(mean=0.0, std=std)
            if self.proj_t.bias is not None:
                self.proj_t.bias.data.zero_()
        for ffn in self.aux_modules.values():
            if isinstance(ffn, nn.ModuleList):
                for _ffn in ffn:
                    _ffn.initialize_weights(std=std)
                continue
            if isinstance(ffn, FeedForward):
                ffn.initialize_weights(std=std)

    def get_intent_embs(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        input_type_ids: Optional[torch.Tensor] = None,
                        encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None,
                        return_attention_scores: Optional[bool] = False,
                        ) -> torch.Tensor:
        if encoder_hidden_states is None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.pad_token_id)
            token_type_ids = None
            if self.input_type_embeddings:
                token_type_ids = input_type_ids
            ret = self.transformer(
                input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=return_attention_scores,
                output_hidden_states=True)

            encoder_hidden_states = ret.last_hidden_state
        # # (Option1) Extract the hidden states of <cls> token (the first token)
        # hidden_states = encoder_hidden_states[-1][:, 0, :]

        # # (Option2) Avg embedding
        # agg = (encoder_hidden_states[-1] * attention_mask.unsqueeze(-1)).sum(dim=1)
        # hidden_states = agg / attention_mask.sum(dim=-1, keepdim=True)

        if self.attention_type == 'none':  # no attention
            hidden_states = encoder_hidden_states[:, 0]
        else:
            # (Option 3) Attention
            if isinstance(self.proj_h, nn.Module):
                encoder_hidden_states = self.proj_h(encoder_hidden_states)
            if self.type_emb_dim > 0:
                assert input_type_ids is not None
                input_type_embs = self.type_embs(input_type_ids)
                if isinstance(self.proj_t, nn.Module):
                    input_type_embs = self.proj_t(input_type_embs)
            if self.attention_type == 'additive':
                if input_type_ids is None or self.type_emb_dim == 0:
                    logits = self.W(self.attention_activation(encoder_hidden_states))
                else:
                    ht = (self.attention_activation(encoder_hidden_states),
                          self.attention_activation(input_type_embs))
                    logits = self.W(torch.cat(ht, dim=-1))
                    logits[~attention_mask] = -float('Inf')
                    attentions = logits.softmax(dim=1)
                    hidden_states = (attentions * encoder_hidden_states).sum(dim=1)
            elif self.attention_type == 'sum':
                if input_type_ids is None or self.type_emb_dim == 0:
                    ht = self.attention_activation(encoder_hidden_states)
                else:
                    ht = self.attention_activation(encoder_hidden_states) \
                        + self.attention_activation(input_type_embs)
                if self.attention is not None:
                    hidden_states, attentions = self.attention(
                        ht, attention_mask=attention_mask, output_attentions=True)
                    attentions = attentions.mean(-1, keepdim=True)  # mean over heads
                else:
                    logits = self.W(ht)
                    logits[~attention_mask] = -float('Inf')
                    attentions = logits.softmax(dim=1)
                    hidden_states = (attentions * encoder_hidden_states).sum(dim=1)
            elif self.attention_type == 'bilinear':
                assert isinstance(input_type_ids, torch.Tensor)
                h = encoder_hidden_states
                t = input_type_embs
                assert h.size() == t.size()
                ht = self.attention_activation(h) * self.attention_activation(t)
                logits = self.W(ht)
            else:
                raise NotImplementedError
        hidden_states = self.extractor(hidden_states)
        if return_attention_scores:
            with torch.no_grad():
                if self.attention_type == 'none':
                    attentions = ret.attentions[-1][:, :, 0].mean(dim=1).unsqueeze(-1)
                    attn_accum = torch.cat([attn[:, :, 0].mean(dim=1).unsqueeze(0)
                                            for attn in ret.attentions], dim=0).sum(dim=0)
                else:
                    attn_accum = torch.cat([attentions.squeeze(-1).unsqueeze(0)]
                                           + [attn[:, :, 0].mean(dim=1).unsqueeze(0)
                                            for attn in ret.attentions], dim=0).sum(dim=0)
                attn_norms = (attentions * encoder_hidden_states).norm(dim=-1)
            return hidden_states, attentions.squeeze(2), attn_norms, attn_accum

        return hidden_states

class SimpleGenClfModel(SimpleGenClfBaseModel):
    def __init__(self,
                 transformer_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 intent_extractor: Optional[Dict[str, Any]] = {},
                 decoder: Optional[Dict[str, Any]] = {},
                 auxiliary: Optional[List[Dict[str, Any]]] = [],
                 primary_loss_type: Optional[str] = 'ce',
                 primary_loss_smoothing_factor: Optional[float] = 0.0,
                 **kwargs) -> None:
        super().__init__(
            transformer_model,
            tokenizer,
            intent_extractor,
            decoder,
            auxiliary,
            **kwargs
        )
        d_model = self.transformer.config.hidden_size

        # Decoder
        dec_conf = decoder
        num_layers = dec_conf.get('num_layers', 1)
        hidden_dims = dec_conf.get('hidden_dims', d_model)
        bias = dec_conf.get('bias', True)
        dropout = dec_conf.get('dropout', 0)
        activations = dec_conf.get('pre_activation', 'tanh')
        self.cross_attention = dec_conf.get('cross_attention', False)
        self.decoder = nn.GRU(d_model, hidden_dims, num_layers, bias,
                              batch_first=True, dropout=dropout,
                              bidirectional=False)
        self.pre_decoder = nn.ModuleList(
            [FeedForward(d_model, 1,
                         self.decoder.input_size,
                         activations, dropout, bias=bias)
             for _ in range(num_layers)])
        self.post_decoder = FeedForward(self.decoder.hidden_size*2, 1,
                                        d_model,
                                        'tanh', dropout, bias=False)
        self.lm_head = nn.Linear(d_model,
                                 self.transformer.get_input_embeddings().num_embeddings,
                                 bias=False)
        self.initialize_decoder()

        # Loss function
        self.primary_output_type = 'discrete'
        if primary_loss_type == 'ce':
            if primary_loss_smoothing_factor == 0:
                self.loss_fn_primary = nn.CrossEntropyLoss(
                    ignore_index=self.pad_token_id,
                    reduction='none')
            else:
                self.loss_fn_primary = LabelSmoothedCELoss(
                    smoothing_factor=primary_loss_smoothing_factor,
                    ignore_index=self.pad_token_id,
                    reduction='none')
        else:
            raise NotImplementedError

    def initialize_decoder(self) -> None:
        std = 1.0
        for ffn in self.pre_decoder:
            ffn.initialize_weights(std=std)
        self.post_decoder.initialize_weights(std=std)
        # Use the default initialization for self.decoder (RNN)
        token_embedder = self.transformer.get_input_embeddings()

        self.lm_head.weight.data.copy_(token_embedder.weight.data)

    def forward(self, batch: Dict,
                label_weights: Optional[Dict[str, torch.Tensor]] = {}) -> Dict[str, Any]:
        input_ids = batch['input_ids']
        input_mask = batch.get('input_mask', input_ids.ne(-1))
        input_type_ids = batch.get('input_type_ids')
        output_ids = batch.get('output_ids')
        output_mask = batch.get('output_mask', output_ids.ne(-1))
        batch_size = input_ids.size(0)

        token_type_ids = None
        if self.input_type_embeddings and self.type_emb_dim == 0:
            # Specified to encode type IDs with transformer
            # and type embeddings (used in the intent extractor) has 0 dim (=not used).
            token_type_ids = input_type_ids


        ret = self.transformer(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=input_mask,
                               output_hidden_states=True)

        intent_embs = self.get_intent_embs(
            input_ids=input_ids, attention_mask=input_mask,
            input_type_ids=input_type_ids,
            encoder_hidden_states=ret.last_hidden_state)

        outputs = {}
        if output_ids is not None:
            h_0 = torch.cat([ffn(intent_embs).unsqueeze(0) for ffn in self.pre_decoder],
                            dim=0)
            logits = self.calculate_logits(encoder_hidden_states=ret.last_hidden_state,
                                           encoder_attention_mask=input_mask,
                                           decoder_input_ids=output_ids,
                                           h_0=h_0)

            shifted_ids = output_ids[:, 1:].contiguous().view(-1)
            vocab_size = logits.size(-1)
            shifted_logits = logits[:, :-1, :].contiguous().view((-1, vocab_size))
            loss = self.loss_fn_primary(shifted_logits, shifted_ids)
            outputs['loss'] = loss.view((batch_size, -1))
            outputs['n'] = output_ids.ne(self.pad_token_id).sum(dim=1) - 1

        for key in self.aux_modules.keys():
            if f'{key}_ids' not in batch:
                continue
            # Which instances have target texts (some instances don't)
            mask = batch[f'{key}_mask']
            # Target labels (texts)
            target_ids = batch[f'{key}_ids']
            target_mask = target_ids.ne(self.pad_token_id).unsqueeze(-1)

            if self.aux_output_type[key] == 'generation':
                decoder = self.aux_modules[key]
                pre_decoder = self.aux_modules[f'{key}_pre_decoder']
                lm_head = self.aux_modules[f'{key}_lm_head']
                h_0 = torch.cat([ffn(intent_embs[mask]).unsqueeze(0) for ffn in pre_decoder],
                                dim=0)
                # The encoder states are passed below but are not used
                logits = self.calculate_logits(encoder_hidden_states=ret.last_hidden_state,
                                               encoder_attention_mask=input_mask,
                                               decoder_input_ids=target_ids,
                                               h_0=h_0,
                                               decoder=decoder, lm_head=lm_head,
                                               cross_attention=False)
                shifted_ids = target_ids[:, 1:].contiguous().view(-1)
                vocab_size = logits.size(-1)
                shifted_logits = logits[:, :-1, :].contiguous().view((-1, vocab_size))
                loss = self.loss_fn_primary(shifted_logits, shifted_ids)
                loss = loss.view((target_mask.size(0), -1, 1)) * target_mask[:, 1:]
                outputs[f'loss_{key}'] = loss.sum(dim=1) / (target_mask[:, 1:].sum(dim=1)+1e-10)
                continue

            # Make predictions
            if isinstance(self.aux_modules[key], FeedForward):
                scores = self.aux_modules[key](intent_embs[mask])
            elif isinstance(self.aux_modules[key], GILE):
                scores = self.aux_modules[key](intent_embs[mask],
                                               self.aux_label_embedders[key].weight,
                                               targeted=False)

            # Get target values and calculate loss
            if self.aux_output_type[key] == 'cbol':
                if key in self.aux_label_embedders:
                    target_embs = self.aux_label_embedders[key](target_ids)
                else:
                    with torch.no_grad(): # Do not calculate gradients
                        target_embs = self.transformer.get_input_embeddings()(target_ids)
                target = (target_embs * target_mask).sum(dim=1)

                loss_values = self.loss_mse_fn(scores, target)
                outputs[f'loss_{key}'] = loss_values.mean(dim=1, keepdim=True)

            elif self.aux_output_type[key] == 'multilabel':
                # Make binary target vectors
                ## IDs --> binary (batchsize, length, labelvocab)
                target_ids = nn.functional.one_hot(target_ids,
                                                   num_classes=scores.size(1))
                target_ids = target_ids.sum(dim=1)  # (batchsize, labelvocab)
                assert target_ids[:, 2:].max() <= 1
                # Note: the first columum corresponds to padding, the second corresponds to UNK
                loss_values = self.loss_bcel_fn(input=scores[:, 2:],
                                                target=target_ids[:, 2:].float())
                if isinstance(label_weights, dict) and key in label_weights:
                    weights = label_weights[key]
                    weights = weights[2:].unsqueeze(0).repeat(loss_values.size(0), 1)
                    if f'{key}_core_ids' in batch:  # FrameNet
                        target_core_ids = batch[f'{key}_core_ids']
                        target_core_ids = nn.functional.one_hot(target_core_ids,
                                                                num_classes=scores.size(1))
                        target_core_ids = target_core_ids.sum(dim=1)[:, 2:]
                        assert target_core_ids.max() <= 1
                        weights[target_core_ids==1] = 1.0  # set the weight of a core label to 1
                    loss_values = loss_values * weights
                outputs[f'loss_{key}'] = loss_values.mean(dim=1, keepdim=True)
            else:
                raise NotImplementedError

        return outputs

    def calculate_logits(self,
                         encoder_hidden_states: torch.Tensor,
                         encoder_attention_mask: torch.Tensor,
                         decoder_input_ids: torch.Tensor,
                         h_0: torch.Tensor,
                         return_hidden_states: Optional[bool] = False,
                         decoder: Optional[nn.Module] = None,
                         post_decoder: Optional[nn.Module] = None,
                         lm_head: Optional[nn.Module] = None,
                         cross_attention: Optional[bool] = None
    ) -> torch.Tensor:
        if decoder is None:
            decoder = self.decoder
        if post_decoder is None:
            post_decoder = self.post_decoder
        if lm_head is None:
            lm_head = self.lm_head
        if cross_attention is None:
            cross_attention = self.cross_attention
        output_embs = self.transformer.get_input_embeddings()(decoder_input_ids)
        decoder_hidden_states, hidden_states = decoder(output_embs, h_0)
        if cross_attention:
            dot = torch.matmul(encoder_hidden_states, decoder_hidden_states.transpose(1, 2)) # \
                # / math.sqrt(encoder_hidden_states.size(-1))
            dot[~encoder_attention_mask] = -float('Inf')
            attentions = dot.transpose(1, 2).softmax(dim=2)
            ctx = torch.matmul(attentions, encoder_hidden_states)
            h = post_decoder(torch.cat((ctx, decoder_hidden_states), dim=2))
        else:
            h = decoder_hidden_states
        logits = lm_head(h)
        if return_hidden_states:
            return logits, hidden_states
        return logits

    def generate_greedy_top1(self,
                             input_ids: torch.LongTensor,
                             attention_mask: Optional[torch.Tensor] = None,
                             input_type_ids: Optional[torch.Tensor] = None,
                             max_length: Optional[int] = None,
                             pad_token_id: Optional[int] = None,
                             eos_token_id: Optional[int] = None,
                             decoder_start_token_id: Optional[int] = None,
                             **model_kwargs
                             ) -> torch.LongTensor:
        pad_token_id = pad_token_id if pad_token_id is not None else self.transformer.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.transformer.config.eos_token_id
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else self.transformer.config.decoder_start_token_id

        # keep track of which sequences are already finished
        output_ids = input_ids.new(input_ids.shape[0], 1).fill_(decoder_start_token_id)
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = 1

        if attention_mask is None:
            attention_mask = input_ids.ne(pad_token_id)

        token_type_ids = None
        if self.input_type_embeddings:
            token_type_ids = input_type_ids

        ret = self.transformer(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        intent_embs = self.get_intent_embs(
            input_ids=input_ids, attention_mask=attention_mask,
            input_type_ids=input_type_ids,
            encoder_hidden_states=ret.last_hidden_state)
        h = torch.cat([ffn(intent_embs).unsqueeze(0) for ffn in self.pre_decoder],
                      dim=0)
        while True:
            logits, h = self.calculate_logits(encoder_hidden_states=ret.last_hidden_state,
                                              encoder_attention_mask=attention_mask,
                                              decoder_input_ids=output_ids[:, -1].unsqueeze(1),
                                              h_0=h,
                                              return_hidden_states=True)
            # output_mask = output_ids.ne(pad_token_id)
            # output_embs = self.transformer.embeddings(input_ids=output_ids[:, -1].unsqueeze(1))
            # out, h = self.decoder(output_embs, h)
            # logits = self.lm_head(out)
            next_tokens = logits.argmax(dim=-1).flatten()

            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            output_ids = torch.cat([output_ids, next_tokens[:, None]], dim=-1)
            cur_len += 1

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 \
               or (isinstance(max_length, int) and cur_len >= max_length):
                break

        return output_ids

    def generate_event_greedy_top1(self,
                                   key: str,
                                   input_ids: torch.LongTensor,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   input_type_ids: Optional[torch.Tensor] = None,
                                   max_length: Optional[int] = None,
                                   pad_token_id: Optional[int] = None,
                                   eos_token_id: Optional[int] = None,
                                   decoder_start_token_id: Optional[int] = None,
                                   **model_kwargs
                             ) -> torch.LongTensor:
        assert key in {'comet-xNeed', 'comet-xIntent'}
        decoder = self.aux_modules[key]
        pre_decoder = self.aux_modules[f'{key}_pre_decoder']
        lm_head = self.aux_modules[f'{key}_lm_head']

        pad_token_id = pad_token_id if pad_token_id is not None else self.transformer.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.transformer.config.eos_token_id
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else self.transformer.config.decoder_start_token_id

        # keep track of which sequences are already finished
        output_ids = input_ids.new(input_ids.shape[0], 1).fill_(decoder_start_token_id)
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = 1

        if attention_mask is None:
            attention_mask = input_ids.ne(pad_token_id)

        token_type_ids = None
        if self.input_type_embeddings:
            token_type_ids = input_type_ids

        ret = self.transformer(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        intent_embs = self.get_intent_embs(
            input_ids=input_ids, attention_mask=attention_mask,
            input_type_ids=input_type_ids,
            encoder_hidden_states=ret.last_hidden_state)
        h = torch.cat([ffn(intent_embs).unsqueeze(0) for ffn in pre_decoder],
                      dim=0)
        while True:
            # The encoder states are passed below but are not used
            logits, h = self.calculate_logits(encoder_hidden_states=ret.last_hidden_state,
                                              encoder_attention_mask=attention_mask,
                                              decoder_input_ids=output_ids[:, -1].unsqueeze(1),
                                              h_0=h,
                                              decoder=decoder, lm_head=lm_head,
                                              cross_attention=False,
                                              return_hidden_states=True)
            # output_mask = output_ids.ne(pad_token_id)
            # output_embs = self.transformer.embeddings(input_ids=output_ids[:, -1].unsqueeze(1))
            # out, h = self.decoder(output_embs, h)
            # logits = self.lm_head(out)
            next_tokens = logits.argmax(dim=-1).flatten()

            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            output_ids = torch.cat([output_ids, next_tokens[:, None]], dim=-1)
            cur_len += 1

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 \
               or (isinstance(max_length, int) and cur_len >= max_length):
                break

        return output_ids

class SimpleCGenClfModel(SimpleGenClfBaseModel):
    """Continuous generation for the primary output"""
    def __init__(self,
                 transformer_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 intent_extractor: Optional[Dict[str, Any]] = {},
                 decoder: Optional[Dict[str, Any]] = {},
                 auxiliary: Optional[List[Dict[str, Any]]] = [],
                 primary_loss_type: Optional[str] = 'l2',
                 **kwargs) -> None:
        super().__init__(
            transformer_model=transformer_model,
            tokenizer=tokenizer,
            intent_extractor=intent_extractor,
            decoder=decoder,
            auxiliary=auxiliary,
            **kwargs
        )
        d_model = self.transformer.config.hidden_size

        # Decoder
        dec_conf = decoder
        num_layers = dec_conf.get('num_layers', 1)
        hidden_dims = dec_conf.get('hidden_dims', d_model)
        bias = dec_conf.get('bias', True)
        dropout = dec_conf.get('dropout', 0)
        activations = dec_conf.get('pre_activation', 'tanh')
        self.decoder = nn.GRU(d_model, hidden_dims, num_layers, bias,
                              batch_first=True, dropout=dropout,
                              bidirectional=False)
        self.pre_decoder = nn.ModuleList(
            [FeedForward(d_model, 1,
                         self.decoder.input_size,
                         activations, dropout, bias=bias)
             for _ in range(num_layers)])
        vocab_size = self.transformer.config.vocab_size
        self.decoder_target_embeddings = nn.Embedding(vocab_size, d_model,
                                                      self.pad_token_id)
        self.initialize_decoder()

        # Loss function
        self.primary_output_type = 'continous'
        self.primary_loss_type = primary_loss_type


    def initialize_decoder(self) -> None:
        std = 1.0
        for ffn in self.pre_decoder:
            ffn.initialize_weights(std=std)
        # Use the default initialization for self.decoder (RNN)
        self.decoder_target_embeddings.load_state_dict(
            self.transformer.embeddings.word_embeddings.state_dict()
        )

    def forward(self, batch: Dict) -> Dict[str, Any]:
        input_ids = batch['input_ids']
        input_mask = batch.get('input_mask', input_ids.ne(-1))
        output_ids = batch.get('output_ids')
        output_mask = batch.get('output_mask', output_ids.ne(-1))
        batch_size = input_ids.size(0)

        ret = self.transformer(input_ids=input_ids,
                               attention_mask=input_mask,
                               output_hidden_states=True)

        intent_embs = self.get_intent_embs(
            input_ids=input_ids, attention_mask=input_mask,
            encoder_hidden_states=ret.last_hidden_state)

        outputs = {}
        if output_ids is not None:
            h_0 = torch.cat([ffn(intent_embs).unsqueeze(0) for ffn in self.pre_decoder],
                            dim=0)
            output_embs = self.transformer.embeddings(input_ids=output_ids)

            if self.primary_loss_type == 'l2':
                with torch.no_grad():
                    target_embs = self.decoder_target_embeddings(output_ids)
                shifted_out = output_embs[:, :-1]
                shifted_tgt = target_embs[:, 1:]
                loss = (shifted_out - shifted_tgt).square().mean(dim=-1) * output_mask[:, 1:]
                outputs['loss'] = loss
                outputs['n'] = output_mask.sum(dim=1) - 1
            else:
                raise NotImplementedError

        for key in self.aux_modules.keys():
            if f'{key}_ids' not in batch:
                continue

            # Make predictions
            mask = batch[f'{key}_mask']
            scores = self.aux_modules[key](intent_embs[mask])

            # Get target values
            target_ids = batch[f'{key}_ids']
            target_mask = target_ids.ne(self.pad_token_id).unsqueeze(-1)
            with torch.no_grad(): # Do not calculate gradients for the target
                target_embs = self.transformer.embeddings(target_ids)
                target_cbow = (target_embs * target_mask).sum(dim=1)

            # Calculate loss
            loss_values = self.loss_mse_fn(scores, target_cbow)
            outputs[f'loss_{key}'] = loss_values.mean(dim=1, keepdim=True)

        return outputs
