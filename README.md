# PyTorch Sketch-RNN

The goal of this repositority is to provide an accurate PyTorch implementation of the [Sketch-RNN model](https://arxiv.org/abs/1704.03477) from Ha & Eck (2017). The [official implementation](https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/README.md) is written in TensorFlow, provided through the magenta library.

## Existing pytorch repo

There is an [existing PyTorch implementation](https://github.com/alexis-jacq/Pytorch-Sketch-RNN) of Sketch-RNN provided by Alexis Jacq. However, Jacq's model deviates from the original in a few important ways. The goal here is to fix these discrepancies for better reproducibility. The important ingredients are as follows:
1. Parameter initialization. The intialization of weights & biases is important, especially for recurrent LSTM weights, which use a special form of orthogonal initialization.
2. Recurrent dropout. Jacq's implementation does not use dropout of any kind (`dropout=p` is passed to `nn.LSTM`, but this has no effect for a single-layer lstm). Here, I take care to implement the recurrent dropout technique from Sketch-RNN.
3. Layer normalization. My custom LSTM cells implement layer normalization exactly as the official repo.
4. HyperLSTM. I have also implemented the HyperLSTM model used for the Sketch-RNN decoder.

## Development notes
1. Encoder LSTM: I have not yet implemented recurrent dropout and layer normalization for the bi-directional encoder LSTM.
2. Input/output dropout: The magenta library offers optional input/output dropout for the decoder LSTM, although they were not used in the Sketch-RNN paper. I have not implemented either.
3. Omniglot: There is some code in `sketch_rnn.omniglot` for training with Omniglot character drawings. This code has various dependencies from my own private libraries and it can be ignored.