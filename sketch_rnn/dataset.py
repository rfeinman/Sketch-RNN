import os
import six
import requests
import numpy as np
import torch

from . import utils


def load_stroke_data(data_dir, hps):
    """Loads the .npz file, and splits the set into train/valid/test."""

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.

    if isinstance(hps.data_set, list):
        datasets = hps.data_set
    else:
        datasets = [hps.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    for dataset in datasets:
        if data_dir.startswith('http://') or data_dir.startswith('https://'):
            data_filepath = '/'.join([data_dir, dataset])
            print('Downloading %s' % data_filepath)
            response = requests.get(data_filepath)
            data = np.load(six.BytesIO(response.content), encoding='latin1')
        else:
            data_filepath = os.path.join(data_dir, dataset)
            data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        print('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']), dataset))

    if train_strokes is None:
        train_strokes = data['train']
        valid_strokes = data['valid']
        test_strokes = data['test']
    else:
        train_strokes = np.concatenate((train_strokes, data['train']))
        valid_strokes = np.concatenate((valid_strokes, data['valid']))
        test_strokes = np.concatenate((test_strokes, data['test']))

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len)))

    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)
    # overwrite the hps with this calculation.
    hps.max_seq_len = max_seq_len
    print('hps.max_seq_len %i.' % hps.max_seq_len)

    return train_strokes, valid_strokes, test_strokes



class SketchRNNDataset:
    def __init__(self,
                 strokes,
                 max_seq_length=250,
                 scale_factor=None,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.limit = limit # clamp x-y offsets to range (-limit, limit)
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.preprocess(strokes) # list of drawings in stroke-3 format, sorted by size
        self.normalize(scale_factor)

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_seq_length points."""
        raw_data = []
        seq_len = []
        count_data = 0
        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_seq_length):
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                raw_data.append(data)
                seq_len.append(len(data))
        seq_len = np.array(seq_len)  # nstrokes for each sketch
        idx = np.argsort(seq_len)
        self.strokes = []
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])
        print("total images <= max_seq_len is %d" % count_data)

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j,0])
                data.append(self.strokes[i][j,1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        data = np.copy(data)
        data[:,0] *= x_scale
        data[:,1] *= y_scale
        return data

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        data = self.strokes[idx]
        data = self.random_scale(data)
        if self.augment_stroke_prob > 0:
            data = utils.augment_strokes(np.copy(data), self.augment_stroke_prob)
        return data

    # ---- methods for batch collate ----

    def pad_batch(self, batch):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        max_len = self.max_seq_length
        start_stroke_token = [0, 0, 1, 0, 0]
        batch_size = len(batch)
        result = np.zeros((batch_size, max_len + 1, 5), dtype=float)
        for i in range(batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = start_stroke_token[2]  # setting S_0 from paper.
            result[i, 0, 3] = start_stroke_token[3]
            result[i, 0, 4] = start_stroke_token[4]

        return result

    def collate_fn(self, batch):
        lengths = [len(seq) for seq in batch]
        batch = self.pad_batch(batch)
        batch = torch.from_numpy(batch).float()
        lengths = torch.tensor(lengths, dtype=torch.long) # Tensor[nstk]
        return batch, lengths