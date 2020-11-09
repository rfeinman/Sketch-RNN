import torch

__all__ = ['seq2seq_step']


def seq2seq_step(model, src, trg, src_lengths=None):
    max_len = model.max_len

    # model forward
    src = src[:,1:max_len+1] # remove sos
    trg_in = trg[:,:max_len]
    params, z_mean, z_logvar = model._forward(src, trg_in, src_lengths)

    # prepare targets
    trg = trg[:,1:max_len+1]
    x, v_onehot = trg.split([2,3], -1)
    assert torch.all(v_onehot.sum(-1) == 1)
    v = v_onehot.argmax(-1)

    # compute losses
    loss_kl = model.loss_kl(z_mean, z_logvar)
    loss_draw = model.loss_draw(x, v, params)
    loss = loss_kl + loss_draw

    return loss