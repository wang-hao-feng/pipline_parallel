import os
import gc
import time
import random
import numpy as np
from tqdm import tqdm
from itertools import cycle
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from parse_args import parse_args
from dataset import dataset_regisiter, collate_fn_regisiter
from prompt import prompt_regisiter
from prompt_fn import prompt_fn_regisiter, image_token_prompt_regisiter
from pp_models import train_model_regisiter
from pp_utils import save_training, save_splited_model
from loss_fn import loss_fn_regisiter
from lr_scheduler import LinearWarmupCosineSchedule
from pipeline_schedule import Schedule1F1BWithoutMergeChunk

import psutil

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_mem():
    mem = psutil.virtual_memory()
    return mem.used / 1024**3

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

if __name__ == "__main__":
    args, model_kwargs, dataset_kwargs, loss_fn_kwargs = parse_args()

    set_seed(args.seed)

    num_workers = 6

    if args.task == 'contrastive':
        assert args.micro_batch_size % (args.negative_num + 2) == 0

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group(rank=rank, world_size=world_size, backend='nccl')
    pp_group = dist.new_group()
    stage_index = rank
    num_stages = world_size

    model, processor, ignore_token_id = train_model_regisiter[args.model](checkpoint_path=os.path.expanduser(args.model_path), 
                                                                          stage=stage_index, 
                                                                          world_size=world_size, 
                                                                          **model_kwargs)
    model.frozen()

    train_dataset = dataset_regisiter[args.dataset](path=os.path.expanduser(args.dataset_path), split='train', **dataset_kwargs, rank=rank, world_size=world_size)
    val_dataset = dataset_regisiter[args.dataset](path=os.path.expanduser(args.dataset_path), split='val', **dataset_kwargs, rank=rank, world_size=world_size)
    collate_fn = partial(collate_fn_regisiter[args.task], 
                         prompt=prompt_regisiter[args.text_prompt], 
                         model_name=args.model.split('-')[0], 
                         max_token_num=args.max_token_num, 
                         prompt_fn=prompt_fn_regisiter[args.prompt_fn], 
                         image_token_prompt_fn=image_token_prompt_regisiter[args.image_token_prompt], 
                         processor=processor, 
                         rank=rank, 
                         world_size=world_size)
    train_loader = infinite_loader(DataLoader(dataset=train_dataset, 
                                              batch_size=int(args.batch_size // (args.negative_num + 2)) if args.task == 'contrastive' or 'caption' in args.dataset else args.batch_size, 
                                              shuffle=True, 
                                            #   num_workers=num_workers, 
                                              collate_fn=collate_fn, 
                                              drop_last=True))
    # train_loader = cycle(train_loader)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=int(args.batch_size // (args.negative_num + 2)) if args.task == 'contrastive' or 'caption' in args.dataset else args.batch_size, 
                            shuffle=False, 
                            # num_workers=num_workers, 
                            collate_fn=collate_fn, 
                            drop_last=True)

    train_param = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(train_param, lr=args.learning_rate, weight_decay=args.weight_decay) if len(train_param) > 0 else None
    lr_scheduler = LinearWarmupCosineSchedule(optimizer, 
                                              warmup_steps=args.warmup_steps, 
                                              total_steps=args.total_steps, 
                                              min_lr=args.min_lr, 
                                              init_lr=args.learning_rate, 
                                              warm_up_start_lr=args.warm_up_start_lr) if optimizer else None

    with autocast(device_type='cuda', dtype=next(model.parameters()).dtype):
        stage = PipelineStage(
            model,
            stage_index,
            num_stages,
            device,
            input_args=model.get_example_input(micro_batch_size=args.micro_batch_size, max_token_num=args.max_token_num),
            output_args=model.get_example_output(micro_batch_size=args.micro_batch_size, max_token_num=args.max_token_num), 
            group=pp_group
        )

        n_microbatches=int(args.batch_size // args.micro_batch_size)
        loss_fn = partial(loss_fn_regisiter[args.task], ignore_token_id=ignore_token_id, n_microbatches=n_microbatches, total_steps=args.total_steps, warm_up_steps=args.warmup_steps, **loss_fn_kwargs)

        train_schedule = Schedule1F1BWithoutMergeChunk(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)
        eval_schedule = Schedule1F1B(stage, n_microbatches=n_microbatches)

        def OneFOneB(schedule:Schedule1F1BWithoutMergeChunk|Schedule1F1B, inputs, labels=None):
            if rank == 0:
                inputs = [i.to(device) for i in inputs]
                schedule.step(*inputs)
                return None, []
            elif rank < world_size - 1 or labels is None:
                outputs = schedule.step()
                return outputs, []
            else:
                losses = []
                output = schedule.step(target=labels.to(device), losses=losses)
                return output, losses

        def sed2hour(second):
            return f'{int(second // 3600):03d}:{int((second // 60) % 60):02d}:{int(second % 60):02d}'

        def get_last_time(step, total_step_time):
            average_time = total_step_time / (step+1)
            last_time = average_time * (args.total_steps - step - 1)
            return sed2hour(last_time)

        step_start_time = time.time()
        step = 0
        # for step in range(args.total_steps):
        for inputs, labels in train_loader:
            if step >= args.total_steps:
                break
            if step % args.accumulate_step == 0:
                optimizer.zero_grad()
            outputs, losses = OneFOneB(train_schedule, inputs, labels)
            del inputs
            del labels
            if step % args.accumulate_step == 0:
                optimizer.step()
                lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            dist.barrier(group=pp_group)
            if rank == world_size - 1:
                loss = sum(losses) / len(losses)
                print(f'Step: {step+1:05d}/{args.total_steps:05d} Learning rate: {lr:.7f} Loss: {loss.item():.4f} Time: {sed2hour(step_time)}<{get_last_time(step, step_time)}')
            del outputs
            for l in losses:
                del l
            del losses
            gc.collect()
            # print(f'rank{rank} gpu:{torch.cuda.max_memory_allocated(device=device) / 1024**3}G')
            if (step + 1) % args.save_steps == 0 and step > args.min_save_steps:
            # if step % args.save_steps == 0:
                save_path = os.path.expanduser(os.path.join(args.result_path, str(step+1)))
                if rank == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                dist.barrier(group=pp_group)
                save_training(model=model,
                              optimizer=optimizer, 
                              scheduler=lr_scheduler, 
                              step=(step+1),  
                              loss=loss.item() if rank == world_size-1 else 0, 
                              path=save_path, 
                              rank=rank, 
                              world_size=world_size, 
                              group=pp_group)
                step_start_time = time.time()
            if (step + 1) %  args.eval_steps == 0 and step > 0:
            # if step %  args.eval_steps == 0:
                evaluate_start_time = time.time()
                total_loss = []
                stage.submod = stage.submod.eval()
                with torch.no_grad():
                    for i, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f'evaluate{step // args.eval_steps}'):
                        if i > 50:
                            break
                        optimizer.zero_grad()
                        outputs, _ = OneFOneB(eval_schedule, inputs)
                        if rank == world_size - 1:
                            outputs = outputs[0]
                            outputs = outputs.view(*((n_microbatches, -1)+outputs.shape[1:]))
                            labels = labels.view(*((n_microbatches, -1)+labels.shape[1:])).to(device)
                            losses = [loss_fn(outputs[i], labels[i], update=False) for i in range(n_microbatches)]
                            total_loss += losses
                        del outputs
                        # dist.barrier(group=pp_group)
                if rank == world_size - 1:
                    evaluate_end_time = time.time()
                    step_start_time = evaluate_end_time
                    print('-'*15+'Eval'+'-'*15)
                    print(f'Step: {step+1}/{args.total_steps} Loss: {sum(total_loss) / len(total_loss)} Time:{sed2hour(evaluate_end_time - evaluate_start_time)}')
                    print('-'*30)
                stage.submod = stage.submod.train()
            step += 1
            dist.barrier(group=pp_group)
        if rank == 0:
            dist.destroy_process_group(group=pp_group)