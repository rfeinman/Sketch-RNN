import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from .rnn import _cell_types, LSTMLayer
from .param_layer import ParameterLayer
from .objective import KLLoss, DrawingLoss

__all__ = ['SketchRNN', 'model_step']


def pack_sequence(seq, lengths):
    pack = rnn_utils.pack_padded_sequence(
        seq, lengths, batch_first=True, enforce_sorted=False)
    return pack

class Encoder(nn.Module):
    def __init__(self, hidden_size, z_size):
        super().__init__()
        self.rnn = nn.LSTM(5, hidden_size, bidirectional=True, batch_first=True)
        self.output = nn.Linear(2*hidden_size, 2*z_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        nn.init.normal_(self.output.weight, 0., 0.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_sequence(x, lengths)
        _, (x, _) = self.rnn(x) # [2,batch,hid]
        x = x.permute(1,0,2).flatten(1).contiguous() # [batch,2*hid]
        z_mean, z_logvar = self.output(x).chunk(2, 1)
        z = z_mean + torch.exp(0.5*z_logvar) * torch.randn_like(z_logvar)
        return z, z_mean, z_logvar


class SketchRNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        # check inputs
        assert hps.enc_model in ['lstm', 'layer_norm', 'hyper']
        assert hps.dec_model in ['lstm', 'layer_norm', 'hyper']
        if hps.enc_model in ['layer_norm', 'hyper']:
            raise NotImplementedError('LayerNormLSTM and HyperLSTM not yet '
                                      'implemented for bi-directional encoder.')
        cell_init = _cell_types[hps.dec_model]
        # encoder modules
        self.encoder = Encoder(hps.enc_rnn_size, hps.z_size)
        # decoder modules
        cell = cell_init(5+hps.z_size, hps.dec_rnn_size, r_dropout=hps.r_dropout)
        self.decoder = torch.jit.script(LSTMLayer(cell, batch_first=True))
        self.init = nn.Linear(hps.z_size, cell.state_size)
        self.param_layer = ParameterLayer(hps.dec_rnn_size, k=hps.num_mixture)
        # loss modules
        self.loss_kl = KLLoss(
            hps.kl_weight,
            eta_min=hps.kl_weight_start,
            R=hps.kl_decay_rate,
            kl_min=hps.kl_tolerance)
        self.loss_draw = DrawingLoss(mask_padding=hps.mask_loss)
        self.max_len = hps.max_seq_len
        self.reset_parameters()

    def reset_parameters(self):
        self.cell.reset_parameters()
        self.encoder.reset_parameters()
        nn.init.normal_(self.init.weight, 0., 0.001)
        nn.init.zeros_(self.init.bias)
        self.param_layer.reset_parameters()

    def forward(self, data, lengths=None):
        max_len = self.max_len

        # The target/expected vectors of strokes
        enc_inputs = data[:,1:max_len+1,:]
        # vectors of strokes to be fed to decoder (include dummy value)
        dec_inputs = data[:,:max_len,:]

        # encoder forward
        z, z_mean, z_logvar = self.encoder(enc_inputs, lengths)

        # initialize decoder state
        state = torch.tanh(self.init(z)).chunk(2, dim=-1)

        # decoder forward
        z_rep = z[:,None].expand(-1,max_len,-1)
        dec_inputs = torch.cat((dec_inputs, z_rep), dim=-1)
        output, _ = self.decoder(dec_inputs, state)

        # mixlayer outputs
        params = self.param_layer(output)

        return params, z_mean, z_logvar


def model_step(model, data, lengths=None):
    # model forward
    params, z_mean, z_logvar = model(data, lengths)

    # prepare targets
    targets = data[:,1:model.max_len+1,:]
    x, v_onehot = targets.split([2,3], -1)
    assert torch.all(v_onehot.sum(-1) == 1)
    v = v_onehot.argmax(-1)

    # compute losses
    loss_kl = model.loss_kl(z_mean, z_logvar)
    loss_draw = model.loss_draw(x, v, params, lengths=lengths)
    loss = loss_kl + loss_draw

    return loss
