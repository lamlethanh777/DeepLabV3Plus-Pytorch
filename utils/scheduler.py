from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


class WarmupPolyLR(_LRScheduler):
    """Poly learning rate scheduler with warmup phase.
    
    Args:
        optimizer: Wrapped optimizer.
        max_iters: Maximum number of training iterations.
        warmup_iters: Number of warmup iterations. Default: 1000.
        power: Power for polynomial decay. Default: 0.9.
        min_lr: Minimum learning rate. Default: 1e-6.
    """
    def __init__(self, optimizer, max_iters, warmup_iters=1000, power=0.9, 
                 last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_iters
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            return [max(base_lr * (1 - (self.last_epoch - self.warmup_iters) / 
                                   (self.max_iters - self.warmup_iters)) ** self.power, 
                       self.min_lr)
                   for base_lr in self.base_lrs]