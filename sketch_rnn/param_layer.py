import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixLayer(nn.Module):
    def __init__(self, input_size, k, d=2, reg_cov=0.):
        super().__init__()
        self.linear = nn.Linear(input_size, k + 2*k*d + k + 3)
        self.k = k
        self.d = d
        self.reg_cov = reg_cov
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, T=1):
        x = self.linear(x)

        # unpack variables: [mix_probs, means, scales, corrs, endings]
        d, k = self.d, self.k
        splits = [k, k*d, k*d, k, 3]
        mix_logits, means, scales, corrs, v_logits = x.split(splits, -1)
        mix_logp = F.log_softmax(mix_logits, -1) # [n,t,k]
        scales = torch.exp(scales) # [n,t,k,d]
        corrs = torch.tanh(corrs) # [n,t,k]
        v_logp = F.log_softmax(v_logits, -1) # [n,t,3]

        # reshape [...,k*d] -> [...,k,d]
        means = means.reshape(*means.shape[:-1], -1, 2)
        scales = scales.reshape(*scales.shape[:-1], -1, 2)

        # add tikhonov regularization
        if self.reg_cov > 0:
            raise NotImplementedError

        if T != 1:
            v_logp = F.log_softmax(v_logp/T, -1)
            mix_logp = F.log_softmax(mix_logp/T, -1)
            scales = scales * math.sqrt(T)

        return mix_logp, means, scales, corrs, v_logp