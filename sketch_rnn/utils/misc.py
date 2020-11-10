"""
SketchRNN data loading and image manipulation utilities.
"""
import numpy as np
import torch


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        raise Exception('input must be a tensor or ndarray.')
    return x.float()