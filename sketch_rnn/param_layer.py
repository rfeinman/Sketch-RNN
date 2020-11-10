import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ParameterLayer']


class ParameterLayer(nn.Module):
    def __init__(self, input_size, k, d=2):
        super().__init__()
        self.linear = nn.Linear(input_size, k + 2*k*d + k + 3)
        self.splits = [k, k*d, k*d, k, 3]
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, T=1):
        # linear forward
        x = self.linear(x)

        # unpack variables & apply activation
        mix_logits, means, scales, corrs, v_logits = x.split(self.splits, -1)
        mix_logp = F.log_softmax(mix_logits, -1) # [...,k]
        scales = torch.exp(scales) # [...,k*d]
        corrs = torch.tanh(corrs) # [...,k]
        v_logp = F.log_softmax(v_logits, -1) # [...,3]

        # reshape [...,k*d] -> [...,k,d]
        means = means.reshape(*means.shape[:-1], -1, 2)
        scales = scales.reshape(*scales.shape[:-1], -1, 2)

        if T != 1:
            v_logp = F.log_softmax(v_logp/T, -1)
            mix_logp = F.log_softmax(mix_logp/T, -1)
            scales = scales * math.sqrt(T)

        return mix_logp, means, scales, corrs, v_logp