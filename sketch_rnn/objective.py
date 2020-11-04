import torch
import torch.nn as nn


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