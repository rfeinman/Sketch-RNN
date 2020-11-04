import numpy as np
import torch
import torch.nn as nn


# ---- KL Divergence loss ----

def kl_divergence(mean, logvar):
    kl = -0.5 * (1 + logvar - mean ** 2 - torch.exp(logvar))
    kl = kl.mean(0).sum()
    return kl

class KLLoss(nn.Module):
    """
    Good default value is eta_decay=1e-5
    """
    def __init__(self, kl_weight=1., eta_min=0.01, R=0.99995, kl_min=0.):
        super().__init__()
        self.kl_weight = kl_weight
        self.factor = 1. - eta_min
        self.R = R
        self.kl_min = kl_min

    @property
    def weight(self):
        eta_step = 1. - self.factor
        weight = self.kl_weight * eta_step
        if self.training:
            self.factor *= self.R
        return weight

    def forward(self, z_mean, z_logvar):
        loss = kl_divergence(z_mean, z_logvar)
        loss = loss.clamp(self.kl_min, float('inf'))
        loss = self.weight * loss
        return loss


# ---- GMM Loss ----

def mvn_log_prob(x, means, scales, corrs):
    x_diff = x.unsqueeze(-2) - means # (...,k,d)
    Z1 = torch.sum(x_diff**2/scales**2, -1) # (...,k)
    Z2 = 2*corrs*torch.prod(x_diff,-1)/torch.prod(scales,-1) # (...,k)
    mvn_logprobs1 = -(Z1-Z2)/(2*(1-corrs**2)) # (...,k)
    mvn_logprobs2 = -torch.log(2*np.pi*torch.prod(scales,-1)*torch.sqrt(1-corrs**2)) # (...,k)
    mvn_logprobs = mvn_logprobs1 + mvn_logprobs2 # (...,k)

    return mvn_logprobs

class GMMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params):
        mix_logp, means, scales, corrs = params
        mvn_logp = mvn_log_prob(x, means, scales, corrs) # [...,k]
        logp = torch.logsumexp(mix_logp + mvn_logp, dim=-1) # [...]
        losses = -logp
        return losses