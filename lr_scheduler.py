import math
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupCosineSchedule(LambdaLR):
    def __init__(self, 
                 optimizer, 
                 warmup_steps:int, 
                 total_steps:int, 
                 min_lr:float, 
                 init_lr:float, 
                 warm_up_start_lr:float=0, 
                 last_epoch:int=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_start_lr = warm_up_start_lr
        super(LinearWarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        lr = self.init_lr
        if step < self.warmup_steps:
            lr = self.warmup_start_lr + float(step) / float(max(1.0, self.warmup_steps)) * (self.init_lr - self.warmup_start_lr)
        else:
            step = step - self.warmup_steps
            max_step = self.total_steps - self.warmup_steps
            lr = (self.init_lr - self.min_lr) * 0.5 *(1 + math.cos(math.pi * step / max_step)) + self.min_lr
        return lr / self.init_lr