# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SketchRNN data loading and image manipulation utilities."""
import warnings
import random
import numpy as np


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def scale_bound(stroke, average_dimension=10.0):
    """Scale an entire image to be less than a certain size."""
    # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
    # modifies stroke directly.
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i; break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
                int(round(sample_strokes[j][0] * factor)),
                int(round(sample_strokes[j][1] * factor)),
                int(sample_strokes[j][2]),
                int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


class HParams():
    def __init__(self):
        # dataset
        self.data_set = ['cat.npz']
        # model params
        self.max_seq_len = 200
        self.enc_model = 'lstm'
        self.dec_model = 'layer_norm'
        self.enc_rnn_size = 256
        self.dec_rnn_size = 512
        self.z_size = 128
        self.num_mixture = 20
        self.r_dropout = 0.1
        #self.input_dropout = 0.0  # Not recommended
        #self.output_dropout = 0.0  # Not recommended
        # loss params
        self.kl_weight = 0.5
        self.kl_weight_start = 0.01 # eta_min
        self.kl_tolerance = 0.2 # kl_min
        self.kl_decay_rate = 0.99995 # R
        # training params
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.0
        self.batch_size = 100
        # data augmentation params
        self.random_scale_factor = 0.15
        self.augment_stroke_prob = 0.10
