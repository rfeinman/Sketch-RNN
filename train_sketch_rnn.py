import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sketch_rnn.utils import HParams, AverageMeter
from sketch_rnn.dataset import SketchRNNDataset, load_strokes, collate_drawings
from sketch_rnn.model import SketchRNN, model_step



def train_epoch(model, data_loader, optimizer, scheduler, device,
                grad_clip=None):
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(data_loader.dataset)) as progress_bar:
        for data, lengths in data_loader:
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            lengths = lengths + 2 # add 2 for SOS and EOS tokens
            # training step
            optimizer.zero_grad()
            loss = model_step(model, data, lengths)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            # update loss meter and progbar
            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg)
            progress_bar.update(data.size(0))

    return loss_meter.avg


@torch.no_grad()
def eval_epoch(model, data_loader, device):
    model.eval()
    loss_meter = AverageMeter()
    for data, lengths in data_loader:
        data = data.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        lengths = lengths + 2 # add 2 for SOS and EOS tokens
        loss = model_step(model, data, lengths)
        loss_meter.update(loss.item(), data.size(0))
    return loss_meter.avg


def train_sketch_rnn(args, hps=None):
    if hps is None:
        hps = HParams()
    torch.manual_seed(884)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    # initialize train and val datasets
    train_strokes, valid_strokes, test_strokes = load_strokes(args.data_dir, hps)
    train_data = SketchRNNDataset(
        train_strokes,
        max_len=hps.max_seq_len,
        random_scale_factor=hps.random_scale_factor,
        augment_stroke_prob=hps.augment_stroke_prob
    )
    val_data = SketchRNNDataset(
        valid_strokes,
        max_len=hps.max_seq_len,
        scale_factor=train_data.scale_factor,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0
    )

    # initialize data loaders
    collate_fn = lambda x : collate_drawings(x, hps.max_seq_len)
    train_loader = DataLoader(
        train_data,
        batch_size=hps.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=hps.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )

    # model & optimizer
    model = SketchRNN(hps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hps.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, hps.lr_decay)

    for i in range(args.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, hps.grad_clip)
        val_loss = eval_epoch(model, val_loader, device)
        print('Epoch %0.3i, Train Loss: %0.4f, Valid Loss: %0.4f' %
              (i+1, train_loss, val_loss))
        time.sleep(0.5) # avoids progress bar issue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    group1 = parser.add_argument_group('script arguments')
    group1.add_argument('--data_dir', type=str,
                        default='/misc/vlgscratch4/LakeGroup/Reuben/Sketch-RNN')
    group1.add_argument('--num_epochs', type=int, default=200)
    group1.add_argument('--num_workers', type=int, default=4)

    group2 = parser.add_argument_group('hyperparameters')
    # model params
    group2.add_argument('--max_seq_len', type=int, default=250)
    group2.add_argument('--enc_model', type=str, default='lstm')
    group2.add_argument('--dec_model', type=str, default='layer_norm')
    group2.add_argument('--enc_rnn_size', type=int, default=256)
    group2.add_argument('--dec_rnn_size', type=int, default=512)
    group2.add_argument('--z_size', type=int, default=128)
    group2.add_argument('--num_mixture', type=int, default=20)
    group2.add_argument('--r_dropout', type=float, default=0.1)
    # KL loss params
    group2.add_argument('--kl_weight', type=float, default=0.5)
    group2.add_argument('--kl_weight_start', type=float, default=0.01) # eta_min
    group2.add_argument('--kl_tolerance', type=float, default=0.2) # kl_min
    group2.add_argument('--kl_decay_rate', type=float, default=0.99995) # R
    group2.add_argument('--mask_loss', action='store_true')
    # training params
    group2.add_argument('--batch_size', type=int, default=100)
    group2.add_argument('--lr', type=float, default=0.001)
    group2.add_argument('--lr_decay', type=float, default=0.9999)
    group2.add_argument('--min_lr', type=float, default=0.00001) # UNUSED
    group2.add_argument('--grad_clip', type=float, default=1.0)
    # dataset & data augmentation params
    group2.add_argument('--data_set', type=str, default='cat.npz')
    group2.add_argument('--random_scale_factor', type=float, default=0.15)
    group2.add_argument('--augment_stroke_prob', type=float, default=0.10)

    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest : getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    args = arg_groups['script arguments']
    hps = arg_groups['hyperparameters']

    train_sketch_rnn(args, hps=hps)
