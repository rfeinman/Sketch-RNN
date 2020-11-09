import os
import numpy as np
import torch
from gns.omniglot.dataset_flat import load_from_orig

from ..dataset import SketchRNNDataset, collate_drawings

__all__ = ['load_omniglot_strokes', 'PairDrawingsDataset', 'collate_pair']


def load_omniglot_strokes(root=None, background=True, min_stk_dist=20.,
                          max_len=52):
    if root is None:
        root = default_root()
    dataset = load_from_orig(root, background=background, canvases=False)
    if min_stk_dist is not None:
        dataset.rm_small_strokes(min_stk_dist)
    strokes = []
    labels = []
    for ex in dataset:
        stk = lines_to_strokes(ex.splines)
        if (max_len is None) or (len(stk) <= max_len):
            strokes.append(stk)
            labels.append(ex.character_id)
    print('Number of examples: {}'.format(len(strokes)))

    return strokes, labels

def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    data = []
    #data.append([0, 0, 0])
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            v = 0 if (i < linelen-1) else 1
            next_pt = [line[i][0], line[i][1], v]
            data.append(next_pt)
    data = np.array(data)
    data[1:, 0:2] -= data[:-1, 0:2]
    return data[1:, :]

def default_root():
    if os.path.exists('/Users/rfeinman'):
        return '/Users/rfeinman/src/NeuralBPL/data'
    elif os.path.exists('/home/feinman'):
        return '/misc/vlgscratch4/LakeGroup/Reuben/Omniglot'
    elif os.path.exists('/home/raf466'):
        return '/scratch/raf466/Omniglot'
    else:
        raise Exception('No default root for this machine. Root directory '
                        'must be provided.')


# ---- Dataset ----

class PairDrawingsDataset(SketchRNNDataset):
    def __init__(self, strokes, labels, **kwargs):
        super().__init__(strokes, **kwargs)
        labels = [c for c,stk in zip(labels, strokes) if len(stk) <= self.max_len]
        labels = [labels[ix] for ix in self.sort_idx]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        other_idx, = torch.where(self.labels == self.labels[idx])
        other_idx = other_idx[other_idx != idx]
        sel = torch.randint(0,len(other_idx),size=())
        idx_trg = other_idx[sel]
        src = super().__getitem__(idx)
        trg = super().__getitem__(idx_trg)

        return src, trg

def collate_pair(data, max_len):
    src, trg = list(zip(*data))
    src_data, src_lengths = collate_drawings(src, max_len)
    trg_data, _ = collate_drawings(trg, max_len)
    return src_data, trg_data, src_lengths