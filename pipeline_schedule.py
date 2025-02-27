import os
import torch
from torch.distributed.pipelining import Schedule1F1B

class Schedule1F1BWithoutMergeChunk(Schedule1F1B):
    def step(self, 
             *args, 
             target=None, 
             losses = None, 
             **kwargs):
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        return None