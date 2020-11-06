import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['KLLoss', 'DrawingLoss']


# ---- KL Divergence loss ----

def kl_divergence(mean, logvar):
    kl = -0.5 * (1 + logvar - mean ** 2 - torch.exp(logvar))
    return kl.mean()

class KLLoss(nn.Module):
    def __init__(self, kl_weight=1., eta_min=0.01, R=0.99995, kl_min=0.):
        super().__init__()
        self.kl_weight = kl_weight
        self.factor = 1. - eta_min
        self.R = R
        self.kl_min = kl_min

    @property
    def weight(self):
        eta = 1. - self.factor
        weight = self.kl_weight * eta
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
    diff = (x.unsqueeze(-2) - means) / scales # [...,k,d]
    z1 = diff.square().sum(-1) # [...,k]
    z2 = 2 * corrs * diff.prod(-1) # [...,k]
    logp1 = - 0.5 * (z1-z2) / (1-corrs**2) # [...,k]
    logp2 = - 0.5 * (1-corrs**2).log() - scales.log().sum(-1) - math.log(2*math.pi)  # [...,k]
    logp = logp1 + logp2 # [...,k]
    return logp

class DrawingLoss(nn.Module):
    def __init__(self, mask_padding=False):
        super().__init__()
        self.mask_padding = mask_padding

    def forward(self, x, v, params, lengths=None):
        mix_logp, means, scales, corrs, v_logp = params
        # mixture losses
        mvn_logp = mvn_log_prob(x, means, scales, corrs) # [batch,step,mix]
        logp = torch.logsumexp(mix_logp + mvn_logp, dim=-1) # [batch,step]
        losses_x = -logp
        # pen action category loss
        losses_v = F.nll_loss(v_logp.flatten(0,1), v.flatten(), reduction='none')
        losses_v = losses_v.reshape(v.shape) # [batch,step]
        # total
        losses = losses_x + losses_v
        if self.mask_padding and (lengths is not None):
            mask = mask_from_lengths(lengths, max_len=x.size(1))
            loss = losses[mask].mean()
        else:
            loss = losses.mean()
        return loss

def mask_from_lengths(lengths, max_len):
    assert len(lengths.shape) == 1, 'lengths shape should be 1 dimensional.'
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    mask = mask.expand(lengths.size(0), -1) < lengths.unsqueeze(1)
    return mask