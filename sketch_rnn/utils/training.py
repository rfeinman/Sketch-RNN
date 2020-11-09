import warnings
import os
import shutil
import numpy as np
import torch
import torch.nn as nn

__all__ = ['AverageMeter', 'ModelCheckpoint']


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelCheckpoint:
    def __init__(self,
                 save_dir,
                 save_freq=5,
                 losses_only=False,
                 best_only=True,
                 tensorboard=False):
        if os.path.exists(save_dir):
            warnings.warn('Save directory already exists! Removing old '
                          'directory contents.')
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        self.model_file = os.path.join(save_dir, 'model.pt')
        self.optimizer_file = os.path.join(save_dir, 'optimizer.pt')
        self.losses_file = os.path.join(save_dir, 'losses.pt')
        self.save_freq = save_freq
        self.losses_only = losses_only
        self.best_only = best_only
        self.losses = np.array([], dtype=np.float32)
        self.val_losses = np.array([], dtype=np.float32)
        self.best = float('inf')
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(save_dir, 'logs')
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    @staticmethod
    def _module(model):
        is_dp = isinstance(model, nn.DataParallel)
        is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
        if is_dp or is_ddp:
            return model.module
        return model

    def __call__(self, epoch, model, optimizer, loss, val_loss):
        # update loss trackers
        self.losses = np.append(self.losses, loss)
        self.val_losses = np.append(self.val_losses, val_loss)
        if self.writer is not None:
            self.writer.add_scalar('loss', loss, epoch)
            self.writer.add_scalar('loss', val_loss, epoch)
            self.writer.flush()

        # save losses
        torch.save({
            'train': torch.from_numpy(self.losses).float(),
            'valid': torch.from_numpy(self.val_losses).float()
            }, self.losses_file)

        if self.losses_only:
            return

        if epoch % self.save_freq == 0:
            model = self._module(model)
            current_loss = self.val_losses[-5:].mean()
            if (not self.best_only) or (current_loss < self.best):
                torch.save(model.state_dict(), self.model_file)
                torch.save(optimizer.state_dict(), self.optimizer_file)
            if current_loss < self.best:
                self.best = current_loss