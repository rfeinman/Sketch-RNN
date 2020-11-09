from typing import Tuple, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LSTMCell', 'LayerNormLSTMCell', 'HyperLSTMCell', 'LSTMLayer',
           'BiLSTMLayer']



def init_orthogonal_(weight, hsize):
    assert weight.size(0) == 4*hsize
    for i in range(4):
        nn.init.orthogonal_(weight[i*hsize:(i+1)*hsize])


# ---- LSTMCell ----

class LSTMCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.,
                 r_dropout=0.1):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.r_dropout = nn.Dropout(r_dropout) if r_dropout > 0 else nn.Identity()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        init_orthogonal_(self.weight_hh, hsize=self.hidden_size)
        nn.init.zeros_(self.bias)

    @property
    def state_size(self):
        return 2 * self.hidden_size

    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        h, c = state
        Wi = torch.mm(x, self.weight_ih.t())
        Wh = torch.mm(h, self.weight_hh.t())
        linear = Wi + Wh + self.bias

        # split and apply activations
        i_gate, f_gate, o_gate, c_cand = linear.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        o_gate = torch.sigmoid(o_gate)
        c_cand = torch.tanh(c_cand)

        # update hidden and cell states
        c = f_gate * c + i_gate * self.r_dropout(c_cand)
        h = o_gate * torch.tanh(c)
        state = h, c

        return h, state



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
        # type: (Tensor) -> Tensor
        x = x.reshape(x.size(0), self.chunks, self.num_units)
        x = F.layer_norm(x, (self.num_units,), None, None, self.eps)
        x = x.reshape(x.size(0), self.chunks*self.num_units)
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
        self.r_dropout = nn.Dropout(r_dropout) if r_dropout > 0 else nn.Identity()
        self.layernorm_h = ChunkLayerNorm(hidden_size, 4)
        self.layernorm_c = nn.LayerNorm(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        init_orthogonal_(self.weight_hh, hsize=self.hidden_size)
        self.layernorm_h.reset_parameters()
        self.layernorm_c.reset_parameters()

    @property
    def state_size(self):
        return 2 * self.hidden_size

    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        h, c = state
        Wi = torch.mm(x, self.weight_ih.t())
        Wh = torch.mm(h, self.weight_hh.t())
        linear = self.layernorm_h(Wi + Wh)

        # split and apply activations
        i_gate, f_gate, o_gate, c_cand = linear.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        o_gate = torch.sigmoid(o_gate)
        c_cand = torch.tanh(c_cand)

        # update hidden and cell states
        c = f_gate * c + i_gate * self.r_dropout(c_cand)
        h = o_gate * torch.tanh(self.layernorm_c(c))
        state = h, c

        return h, state



# ---- HyperLSTMCell ----

class HyperNorm(nn.Module):
    def __init__(self, input_size, embed_size, output_size, bias=True):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_size, embed_size, bias=True),
            nn.Linear(embed_size, output_size, bias=False)
        )
        if bias:
            self.bias_net = nn.Sequential(
                nn.Linear(input_size, embed_size, bias=False),
                nn.Linear(embed_size, output_size, bias=False)
            )
        else:
            self.bias_net = None
        self.embed_size = embed_size
        self.reset_parameters()

    def reset_parameters(self):
        init_gamma = 0.1
        nn.init.constant_(self.scale_net[0].weight, 0.)
        nn.init.constant_(self.scale_net[0].bias, 1.)
        nn.init.constant_(self.scale_net[1].weight, init_gamma/self.embed_size)
        if self.bias_net is not None:
            nn.init.normal_(self.bias_net[0].weight, 0., 0.01)
            nn.init.constant_(self.bias_net[1].weight, 0.)

    def forward(self, x, hyper_out):
        # type: (Tensor, Tensor) -> Tensor
        scale = self.scale_net(hyper_out)
        out = scale * x
        if self.bias_net is not None:
            bias = self.bias_net(hyper_out)
            out = out + bias
        return out


class HyperLSTMCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 forget_bias=1.,
                 r_dropout=0.1,
                 layer_norm=True,
                 hyper_hidden_size=256,
                 hyper_embed_size=32,
                 hyper_r_dropout=0.1):
        super().__init__()
        # hyper LSTM cell
        hyper_init = LayerNormLSTMCell if layer_norm else LSTMCell
        self.hyper_cell = hyper_init(
            input_size,
            hyper_hidden_size,
            forget_bias=forget_bias,
            r_dropout=hyper_r_dropout
        )
        # outer LSTM params
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.r_dropout = nn.Dropout(r_dropout) if r_dropout > 0 else nn.Identity()
        if layer_norm:
            self.layernorm_h = ChunkLayerNorm(hidden_size, 4)
            self.layernorm_c = nn.LayerNorm(hidden_size)
            self.bias = None
        else:
            self.layernorm_h = self.layernorm_c = None
            self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        # hypernorm layers
        def norm_init(use_bias):
            return HyperNorm(hyper_hidden_size, hyper_embed_size, hidden_size, use_bias)
        self.norms_x = nn.ModuleList([
            norm_init(use_bias=False),
            norm_init(use_bias=False),
            norm_init(use_bias=False),
            norm_init(use_bias=False)
        ])
        self.norms_h = nn.ModuleList([
            norm_init(use_bias=True),
            norm_init(use_bias=True),
            norm_init(use_bias=True),
            norm_init(use_bias=True)
        ])
        # misc
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.layer_norm = layer_norm
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embed_size = hyper_embed_size
        self.reset_parameters()

    def reset_parameters(self):
        self.hyper_cell.reset_parameters()
        nn.init.xavier_uniform_(self.weight_ih)
        init_orthogonal_(self.weight_hh, hsize=self.hidden_size)
        if self.layer_norm:
            self.layernorm_h.reset_parameters()
            self.layernorm_c.reset_parameters()
        else:
            nn.init.zeros_(self.bias)
        for norm in self.norms_x:
            norm.reset_parameters()
        for norm in self.norms_h:
            norm.reset_parameters()

    @property
    def state_size(self):
        return 2 * (self.hidden_size + self.hyper_hidden_size)

    # def _apply_hypernorm(self, h_hyper, Wx, Wh):
    #     gates_x = [norm(gate,h_hyper) for norm,gate in zip(self.norms_x, Wx.chunk(4,1))]
    #     gates_h = [norm(gate,h_hyper) for norm,gate in zip(self.norms_h, Wh.chunk(4,1))]
    #     gates = [gx+gh for gx,gh in zip(gates_x, gates_h)]
    #     gates = torch.cat(gates, 1)
    #     return gates

    def _apply_hypernorm(self, h_hyper, Wx, Wh):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        gates_x = Wx.chunk(4,1)
        gates_h = Wh.chunk(4,1)
        gates = torch.jit.annotate(List[Tensor], [])
        i = 0
        for norm in self.norms_x:
            g_in = gates_x[i]
            g_out = norm(g_in, h_hyper)
            gates += [g_out]
            i += 1
        i = 0
        for norm in self.norms_h:
            g_in = gates_h[i]
            g_out = norm(g_in, h_hyper)
            gates[i] += g_out
            i += 1
        gates = torch.cat(gates, 1)
        return gates

    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        h_total, c_total = state
        h, h_hyper = h_total.split((self.hidden_size, self.hyper_hidden_size), 1)
        c, c_hyper = c_total.split((self.hidden_size, self.hyper_hidden_size), 1)

        # hyper lstm cell
        _, (h_hyper, c_hyper) = self.hyper_cell(x, (h_hyper, c_hyper))

        # compute linear gate activations
        Wx = torch.mm(x, self.weight_ih.t())
        Wh = torch.mm(h, self.weight_hh.t())
        gates = self._apply_hypernorm(h_hyper, Wx, Wh)
        if self.layernorm_h is not None:
            gates = self.layernorm_h(gates)
        else:
            gates = gates + self.bias

        # split and apply activations
        i_gate, f_gate, o_gate, c_cand = gates.chunk(4, 1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        o_gate = torch.sigmoid(o_gate)
        c_cand = torch.tanh(c_cand)

        # update hidden and cell states
        c = f_gate * c + i_gate * self.r_dropout(c_cand)
        c_input = self.layernorm_c(c) if (self.layernorm_c is not None) else c
        h = o_gate * torch.tanh(c_input)

        # collect total state
        h_total = torch.cat((h, h_hyper), 1)
        c_total = torch.cat((c, c_hyper), 1)
        state = h_total, c_total

        return h, state



_cell_types = {
    'lstm' : LSTMCell,
    'layer_norm' : LayerNormLSTMCell,
    'hyper' : HyperLSTMCell
}




# ---- LSTM Layer ----

class LSTMLayer(nn.Module):
    def __init__(self,
                 cell,
                 batch_first=False,
                 reverse=False):
        super().__init__()
        self.cell = cell
        self.dim = 1 if batch_first else 0
        self.reverse = reverse
        self.reset_parameters()

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        if self.reverse:
            inputs = torch.flip(inputs, dims=[self.dim])
        inputs = inputs.unbind(dim=self.dim)
        outputs = []
        for t in range(len(inputs)):
            out, state = self.cell(inputs[t], state)
            outputs += [out]
        outputs = torch.stack(outputs, dim=self.dim)
        if self.reverse:
            outputs = torch.flip(outputs, dims=[self.dim])

        return outputs, state


class BiLSTMLayer(nn.Module):
    def __init__(self,
                 cell_f,
                 cell_r,
                 batch_first=False):
        super().__init__()
        self.layer_f = LSTMLayer(cell_f, batch_first)
        self.layer_r = LSTMLayer(cell_r, batch_first, reverse=True)
        self.dim = 1 if batch_first else 0

    def forward(self, inputs, states):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = states
        state_f = torch.jit.annotate(Tuple[Tensor,Tensor], (hx[0], cx[0]))
        state_r = torch.jit.annotate(Tuple[Tensor,Tensor], (hx[1], cx[1]))
        out_f, out_state_f = self.layer_f(inputs, state_f)
        out_r, out_state_r = self.layer_r(inputs, state_r)
        hy = torch.stack((out_state_f[0], out_state_r[0]), 0)
        cy = torch.stack((out_state_f[1], out_state_r[1]), 0)

        out = torch.cat((out_f, out_r), -1)
        out_states = torch.jit.annotate(Tuple[Tensor,Tensor], (hy, cy))

        return out, out_states