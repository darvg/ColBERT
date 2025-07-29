import torch
import contextlib

from colbert.utils.utils import NullContextManager


class MixedPrecisionManager():
    def __init__(self, activated: bool):
        self.activated = activated
        if self.activated:
            # CHANGE: GradScaler is now imported from torch.amp
            self.scaler = torch.GradScaler()

    def context(self):
        # CHANGE: Use torch.amp.autocast and specify the device_type
        if self.activated:
            return torch.autocast(device_type='cuda')
        else:
            return contextlib.nullcontext()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            if scheduler is not None:
                torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if scheduler is not None:
                torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
