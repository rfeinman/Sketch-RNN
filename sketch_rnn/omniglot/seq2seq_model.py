import torch

from ..model import SketchRNN

__all__ = ['Seq2SeqSketchRNN', 'seq2seq_step']


class Seq2SeqSketchRNN(SketchRNN):
    def forward(self, src, trg, src_lengths=None):
        # encoder forward
        z, z_mean, z_logvar = self.encoder(src, src_lengths)
        # initialize decoder state
        state = torch.tanh(self.init(z)).chunk(2, dim=-1)
        # append z to decoder inputs
        z_rep = z[:,None].expand(-1,self.max_len,-1)
        dec_inputs = torch.cat((trg, z_rep), dim=-1)
        # decoder forward
        output, _ = self.decoder(dec_inputs, state)
        # mixlayer outputs
        params = self.param_layer(output)

        return params, z_mean, z_logvar

def seq2seq_step(model, src, trg, src_lengths=None):
    max_len = model.max_len

    # model forward
    src = src[:,1:max_len+1] # remove sos
    trg_in = trg[:,:max_len]
    params, z_mean, z_logvar = model(src, trg_in, src_lengths)

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