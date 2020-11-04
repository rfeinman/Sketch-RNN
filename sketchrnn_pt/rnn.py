import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LSTMCell', 'LayerNormLSTMCell']


class LSTMCell(nn.Module):
    """
    LSTM cell with optional recurrent dropout.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.,
                 r_dropout=0.1):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.r_dropout = nn.Dropout(r_dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.zeros_(self.bias)

    def forward(self, x, state):
        h, c = state
        Wi = torch.mm(x, self.weight_ih.t())
        Wh = torch.mm(h, self.weight_hh.t())
        linear = Wi + Wh + self.bias

        # split and apply activations
        i_gate, f_gate, o_gate, c_new = linear.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        o_gate = torch.sigmoid(o_gate)
        c_new = torch.tanh(c_new)

        # update hidden and cell states
        c = f_gate * c + i_gate * self.r_dropout(c_new)
        h = o_gate * torch.tanh(c)

        return h, c

# ---- LayerNormLSTMCell ----

class ChunkLayerNorm(nn.Module):
    def __init__(self, num_units, chunks, eps=1e-5, affine=True):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.empty(chunks*num_units))
            self.bias = nn.Parameter(torch.empty(chunks*num_units))
        self.num_units = num_units
        self.chunks = chunks
        self.eps = eps
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.chunks, self.num_units)
        x = F.layer_norm(x, (self.num_units,), None, None, self.eps)
        x = x.reshape(*batch_shape, self.chunks*self.num_units)
        if self.affine:
            x = x * self.weight + self.bias
        return x

class LayerNormLSTMCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.,
                 r_dropout=0.1):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.r_dropout = nn.Dropout(r_dropout)
        self.layernorm_h = ChunkLayerNorm(hidden_size, 4)
        self.layernorm_c = nn.LayerNorm(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        for m in [self.layernorm_i, self.layernorm_h, self.layernorm_c]:
            m.reset_parameters()

    def forward(self, x, state):
        h, c = state
        Wi = torch.mm(x, self.weight_ih.t())
        Wh = torch.mm(h, self.weight_hh.t())
        linear = self.layernorm_h(Wi + Wh)

        # split and apply activations
        i_gate, f_gate, o_gate, c_new = linear.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        o_gate = torch.sigmoid(o_gate)
        c_new = torch.tanh(c_new)

        # update hidden and cell states
        c = f_gate * c + i_gate * self.r_dropout(c_new)
        h = o_gate * torch.tanh(self.layernorm_c(c))

        return h, c