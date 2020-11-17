import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import tikhonov_reg2d


__all__ = ['KLLoss', 'DrawingLoss']


# ---- KL Divergence loss ----

def kl_divergence_sn_prior(q_mean, q_logvar):
    """KL with standard normal prior (default)"""
    kl = -0.5 * (1 + q_logvar - q_mean ** 2 - torch.exp(q_logvar))
    return kl.mean()

def kl_divergence(q_mean, q_logvar, p_mean=None, p_logvar=None):
    if p_mean is None and p_logvar is None:
        return kl_divergence_sn_prior(q_mean, q_logvar)
    if p_mean is None: p_mean = torch.zeros_like(q_mean)
    if p_logvar is None: p_logvar = torch.zeros_like(q_logvar)
    kl = p_logvar - q_logvar + \
        (q_logvar.exp() + (q_mean - p_mean)**2) / p_logvar.exp() - 1
    kl = 0.5 * kl
    return kl.mean()

class KLLoss(nn.Module):
    def __init__(self, kl_weight=1., eta_min=0.01, R=0.99995, kl_min=0.):
        super().__init__()
        self.kl_weight = kl_weight
        self.eta_min = eta_min
        self.R = R
        self.kl_min = kl_min
        self.register_buffer('factor', torch.tensor(1-eta_min, dtype=torch.float))

    def reset_parameters(self):
        self.factor.fill_(1-self.eta_min)

    @property
    def weight(self):
        eta = 1. - self.factor.item()
        weight = self.kl_weight * eta
        if self.training:
            self.factor.mul_(self.R)
        return weight

    def forward(self, q_mean, q_logvar, p_mean=None, p_logvar=None):
        if self.kl_weight == 0:
            return torch.tensor(0., device=q_mean.device)
        loss = kl_divergence(q_mean, q_logvar, p_mean, p_logvar)
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
    """
    Parameters
    ----------
    reg_covar : float
        Non-negative regularization added to the diagonal of covariance.
    """
    def __init__(self, reg_covar=1e-6):
        super().__init__()
        self.reg_covar = reg_covar

    def reset_parameters(self):
        pass

    def forward(self, x, v, params):
        # unpack predicted parameters
        mix_logp, means, scales, corrs, v_logp = params
        if self.reg_covar > 0:
            scales, corrs = tikhonov_reg2d(scales, corrs, alpha=self.reg_covar)
        # losses_x: loss wrt pen offset (L_s in equation 9)
        mvn_logp = mvn_log_prob(x, means, scales, corrs) # [batch,step,mix]
        gmm_logp = torch.logsumexp(mix_logp + mvn_logp, dim=-1) # [batch,step]
        losses_x = -gmm_logp
        # losses_v: loss wrt pen state (L_p in equation 9)
        losses_v = F.nll_loss(v_logp.flatten(0,1), v.flatten(), reduction='none')
        losses_v = losses_v.reshape(v.shape) # [batch,step]
        # total average loss
        # padding is masked always for x and only in eval mode for v
        loss_x = losses_x[v!=2].mean()
        loss_v = losses_v.mean() if self.training else losses_v[v!=2].mean()
        loss = loss_x + loss_v

        return loss