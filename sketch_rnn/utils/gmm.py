import torch
import torch.distributions as D

__all__ = ['compute_cov2d', 'sample_gmm']



def compute_cov2d(scales, corrs):
    """
    Compute covariance matrix for 2D multivariate normal

    Parameters
    ----------
    scales: Tensor[...,d]
    corrs: Tensor[...]
    """
    # compute covariances
    cov12 = corrs*torch.prod(scales,dim=-1) # (...,)
    covs = torch.diag_embed(scales**2) # (...,d,d)
    I = torch.diag_embed(torch.ones_like(scales)) # (...,d,d)
    covs = covs + cov12.unsqueeze(-1).unsqueeze(-1)*(1.-I)
    return covs


def sample_gmm(mix_logp, means, scales, corrs):
    covs = compute_cov2d(scales, corrs)
    mix = D.Categorical(mix_logp.exp())
    comp = D.MultivariateNormal(means, covs)
    x = D.MixtureSameFamily(mix, comp).sample() # [1,2]
    return x
