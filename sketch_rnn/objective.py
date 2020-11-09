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
        if self.kl_weight == 0:
            return torch.tensor(0., device=z_mean.device)
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
    """
    Parameters
    ----------
    reg_covar : float
        Non-negative regularization added to the diagonal of covariance.
    """
    def __init__(self, reg_covar=1e-6):
        super().__init__()
        self.reg_covar = reg_covar

    def _tikhonov(self, scales, corrs):
        if self.reg_covar == 0:
            return scales, corrs
        scales_ = torch.sqrt(scales**2 + self.reg_covar)
        corrs_ = corrs * torch.prod(scales, -1) / torch.prod(scales_, -1)
        return scales_, corrs_

    def forward(self, x, v, params):
        # unpack predicted parameters
        mix_logp, means, scales, corrs, v_logp = params
        scales, corrs = self._tikhonov(scales, corrs)
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