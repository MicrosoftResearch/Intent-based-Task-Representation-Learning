from typing import List
from typing import Optional
from typing import Union

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


class FeedForward(nn.Module):
    u"""Feedforward network module."""

    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]],
                 dropout: Optional[Union[float, List[float]]] = 0,
                 bias: Optional[Union[bool, List[bool]]] = True,
                 input_activation: Optional[str] = None,
                 input_dropout: Optional[float] = 0) -> None:
        super().__init__()

        self.input_dropout = nn.Dropout(p=input_dropout)
        self.input_activation = get_activation_func(input_activation)

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        activations = [get_activation_func(act) for act in activations]
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        if not isinstance(bias, list):
            bias = [bias] * num_layers
        assert len(hidden_dims) == num_layers
        assert len(activations) == num_layers
        assert len(dropout) == num_layers
        assert len(bias) == num_layers

        self._activations = nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = [nn.Linear(layer_input_dim, layer_output_dim,
                                   bias=True if not isinstance(bias_, bool) else bias_)
                         for layer_input_dim, layer_output_dim, bias_
                         in zip(input_dims, hidden_dims, bias)]
        self._linear_layers = nn.ModuleList(linear_layers)
        dropout_layers = [nn.Dropout(p=value) for value in dropout]
        self._dropout = nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def initialize_weights(self,
                           mean: Optional[float] = 0,
                           std: Optional[float] = 1.0,
                           identity: Optional[bool] = False) -> None:
        """Initialize linear layers"""
        if identity:
            for module in self._linear_layers:
                assert module.weight.data.size(0) == module.weight.data.size(1)
                assert module.bias is None
                module.weight.data.copy_(torch.eye(module.weight.data.size(1)))
            return
        for module in self._linear_layers:
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_output_dim(self):
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_activation(self.input_dropout(x))
        for layer, activation, dropout in zip(
                self._linear_layers, self._activations, self._dropout
        ):
            out = dropout(activation(layer(out)))
        return out
