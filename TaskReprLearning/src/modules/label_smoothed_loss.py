from typing import Optional
from typing import Tuple

from torch import nn
from torch.nn import functional as F
import torch


class LabelSmoothedCELoss(nn.Module):
    def __init__(self,
                 smoothing_factor: Optional[float] = 0.1,
                 ignore_index: Optional[int] = -100,
                 weight: Optional[torch.Tensor] = None,
                 reduction: Optional[str] = 'mean') -> None:
        super(LabelSmoothedCELoss, self).__init__()

        self.smoothing_factor = smoothing_factor
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        lprobs = F.log_softmax(input, dim=-1)
        return label_smoothed_nll_loss(lprobs, target,
                                       epsilon=self.smoothing_factor,
                                       ignore_index=self.ignore_index,
                                       reduction=self.reduction)

def label_smoothed_nll_loss(lprobs: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: Optional[float] = 0.1,
                            ignore_index: Optional[int] = -100,
                            reduction: Optional[str] = 'mean'
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """From https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/utils.py#L49
    Originally from fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    eps_i = epsilon / lprobs.size(-1)
    loss = ((1.0 - epsilon) * nll_loss + eps_i * smooth_loss).squeeze(-1)

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()

    return loss
