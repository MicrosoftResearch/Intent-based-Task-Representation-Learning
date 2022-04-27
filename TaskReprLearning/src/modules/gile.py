from typing import Optional

from torch import nn
import torch

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


class GILE(nn.Module):
    u"""Nikolaos Pappas and James Henderson. 2019. GILE: A Generalized Input-Label Embedding for Text Classification. Transactions of the Association for Computational Linguistics, 7:139–155, March."""

    def __init__(self,
                 input_dim: int,
                 label_dim: int,
                 joint_dim: int,
                 activation: Optional[str] = 'relu',
                 dropout: Optional[float] = 0,
                 bias: Optional[bool] = True) -> None:
        super().__init__()

        self.activation = get_activation_func(activation)
        self.dropout = nn.Dropout(p=dropout)

        self.proj_h = nn.Linear(input_dim, joint_dim, bias=bias)
        self.proj_l = nn.Linear(label_dim, joint_dim, bias=bias)
        self.proj = nn.Linear(joint_dim, 1, bias=bias)
        self._input_dim = input_dim
        self._label_dim = label_dim
        self._joint_dim = joint_dim
        self._output_dim = 1

    def initialize_weights(self,
                           mean: Optional[float] = 0,
                           std: Optional[float] = 1.0) -> None:
        """Initialize linear layers"""
        for module in [self.proj_h, self.proj_l, self.proj]:
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_output_dim(self):
        return self._output_dim

    def forward(self,
                h: torch.Tensor, l: torch.Tensor,
                targeted: Optional[bool] = True,
                ) -> torch.Tensor:

        h = self.activation(self.proj_h(self.dropout(h)))
        l = self.activation(self.proj_l(self.dropout(l)))

        if targeted:  # return scores only for the given label embeddings
            # assert h.size(0) == l.size(0)  # the first dimension is batch
            # return self.proj(self.dropout(h * l))  # out dim -> (batchsize, 1)
            # # Not tested yet
            raise NotImplementedError
        
        # Return scores for all labels (l is an embedding matrix)
        # h: (batchsize, joint_dim)
        # l: (labelvocab, joint_dim)
        assert len(h.size()) == len(l.size()) == 2
        joint = h.unsqueeze(1) * l.unsqueeze(0)  # (batchsize, labelvocab, joint_dim)
        scores = self.proj(self.dropout(joint))  # (batchsize, labelvocab, 1)
        return scores.squeeze(2)  # (batchsize, labelvocab)
