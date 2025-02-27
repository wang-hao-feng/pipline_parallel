import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler

from safetensors import safe_open
from safetensors.torch import save_model
from accelerate import init_empty_weights

from modules.lora import LoRAWrapper

def load_splited_model(splited_model:nn.Module, path:str, train:bool=False):
    device = f'cuda:{splited_model.stage}' if splited_model.stage != -1 else 'cuda'
    key_map = splited_model.get_key_map_without_last_lora()
    # device = f'cuda:0'
    with open(os.path.join(path, 'model.safetensors.index.json'), encoding='utf-8') as f:
        file = json.load(f)
        weight_map = file['weight_map'] if 'weight_map' in file else file
    splited_model_weight_map = {}
    for name in splited_model.state_dict().keys():
        original_name = key_map[name] if train else name
        if original_name not in weight_map:
            continue
        file_name = weight_map[original_name]
        if file_name not in splited_model_weight_map:
            splited_model_weight_map[file_name] = []
        splited_model_weight_map[file_name].append(name)
    state_dict = splited_model.state_dict()
    for file_name in splited_model_weight_map:
        with safe_open(os.path.join(path, file_name), framework='pt', device=device) as f:
            state_dict.update({
                name: f.get_tensor(key_map[name] if train else name)
                for name in splited_model_weight_map[file_name]
            })
    splited_model.load_state_dict(state_dict, assign=True)
    if train:
        for name, module in splited_model.named_modules():
            if isinstance(module, LoRAWrapper) and 'lora_module' not in name:
                module.init_lora()

    for name, module in splited_model.named_modules():
        for attr in dir(module):
            var = getattr(module, attr)
            if isinstance(var, torch.Tensor) and str(var.device) != device:
                if var.is_meta:
                    print(f'{name}.{attr}')
                setattr(module, attr, var.to(device=device))       

def split_model(cls, config, world_size, path) -> list[nn.Module]:
    with init_empty_weights():
        models = [cls(config=config, stage=stage, world_size=world_size) for stage in range(world_size)]
    for model in tqdm(models, desc='load splited model'):
        load_splited_model(model, path)
    return models

def save_splited_model(module:nn.Module, 
                       path:str, 
                       rank:int, 
                       world_size:int, 
                       group=None):
    output_file = os.path.join(path, f'model-{rank+1:05d}-of-{world_size:05d}.safetensors')
    save_model(module, output_file)
    dist.barrier(group=group)
    if rank == 0:
        weight_map = {}
        for s in range(world_size):
            file_name = f'model-{s+1:05d}-of-{world_size:05d}.safetensors'
            with safe_open(os.path.join(path, file_name), framework='pt', device='cpu') as f:
                weight_map.update({
                    key: file_name
                    for key in f.keys()
                })
        with open(os.path.join(path, 'model.safetensors.index.json'), 'w', encoding='utf-8') as f:
            json.dump({'weight_map': weight_map}, f, ensure_ascii=False)
    dist.barrier(group=group)

def save_training(model:nn.Module, 
                  optimizer:Optimizer, 
                  scheduler:LRScheduler, 
                  step:int, 
                  loss:float, 
                  path:str, 
                  rank:int, 
                  world_size:int, 
                  group=None):
    save_splited_model(model, path, rank, world_size, group=group)
    output_file = os.path.join(path, f'training_info-{rank+1:05d}-of-{world_size:05d}.pth')
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict(), 
        'step': step, 
        'loss': loss
    }, output_file)
    dist.barrier(group=group)