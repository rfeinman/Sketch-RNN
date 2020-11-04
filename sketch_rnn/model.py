import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from .rnn import _cell_types
from .mix_layer import MixLayer



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

    def forward(self, x, lengths):
        x = pack_sequence(x, lengths)
        _, (x, _) = self.rnn(x) # [2,batch,hid]
        x = x.permute(1,0,2).flatten(1).contiguous() # [batch,2*hid]
        z_mean, z_logvar = self.output(x).chunk(2, 1)
        z = z_mean + torch.exp(0.5*z_logvar) * torch.randn_like(z_logvar)
        return z, z_mean, z_logvar


class SketchRNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        # Decoder LSTM
        assert hps.dec_model in ['lstm', 'layer_norm', 'hyper']
        cell_init = _cell_types[hps.dec_model]
        self.cell = cell_init(5, hps.dec_rnn_size, r_dropout=hps.r_dropout)
        self.encoder = Encoder(hps.enc_rnn_size, hps.z_size)
        self.state_init = nn.Sequential(hps.z_size, self.cell.state_size)
        self.mix_layer = MixLayer(hps.dec_rnn_size, k=hps.num_mixture)
        self.max_len = hps.max_seq_len
        self.hps = hps

    def forward(self, data, lengths):
        # The target/expected vectors of strokes
        output_x = data[:,1:self.max_len+1,:]
        # vectors of strokes to be fed to decoder (include dummy value)
        input_x = data[:,:self.max_len,:]

        # encoder forward
        z, z_mean, z_logvar = self.encoder(output_x, lengths)

        # initialize decoder state
        state = self.state_init(z).chunk(2, -1)

        # decoder forward
        input_x = input_x.unbind(1)
        output = []
        for t in range(self.max_len):
            inputs = torch.cat((input_x[t], z), -1)
            out, state = self.cell(inputs, state)
            output.append(out)
        output = torch.stack(output, 1) # [batch,steps,dim]

        # mixlayer outputs
        params = self.mix_layer(output)

        return params
