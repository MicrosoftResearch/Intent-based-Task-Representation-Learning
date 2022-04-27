from typing import Optional

from torch import nn
from torch.nn import functional as F
import torch

import math


class MultiHeadAttentionWithFixedQuery(nn.Module):
    """Multi-head attention layer"""
    def __init__(self,
                 input_size: int,
                 num_attention_heads: int,
                 dropout: Optional[float] = 0,
                 no_transformation: Optional[bool] = False):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = input_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # As we always use the same query, fn_query is just transformation
        self.fn_query = nn.ModuleList([nn.Linear(self.attention_head_size, 1, bias=False)
                                       for _ in range(self.num_attention_heads)])
        if no_transformation:  # do not transform keys and values
            self.fn_key = nn.Identity()
            self.fn_value = nn.Identity()
        else:
            self.fn_key = nn.Linear(input_size, self.all_head_size)
            self.fn_value = nn.Linear(input_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def initialize_normal(self,
                          mean: Optional[float] = 0,
                          std: Optional[float] = 0.02):
        for module in self.fn_query:
            module.weight.data.normal_(mean=0.0, std=std)
        for module in [self.fn_key, self.fn_value]:
            try:
                module.weight.data.normal_(mean=0.0, std=std)
            except:  # fn is Identity
                continue
            if module.bias is None:
                continue
            module.bias.data.zero_()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False
    ):
        batch_size, seq_length = hidden_states.size()[:2]
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        key = self.fn_key(hidden_states).view(*new_shape)
        value = self.fn_value(hidden_states).view(*new_shape)

        attention_scores = torch.cat([W(key[:, :, i]) / math.sqrt(self.attention_head_size)
                                      for i, W in enumerate(self.fn_query)], dim=-1)
        # (batch_size, seq_length, num_attention_heads)
        if attention_mask is not None:
            attention_scores[~attention_mask] = -float('Inf')
        attention_probs = F.softmax(attention_scores, dim=1)

        output = (attention_probs.unsqueeze(-1) * value).sum(1).view(batch_size, -1)
        assert output.size(-1) == self.all_head_size

        if output_attentions:
            return output, attention_probs
        return output
