import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int, 
                 r:int=8, 
                 lora_alpha:int=8, 
                 lora_dropout:float=0.1,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty((input_dim, self.r)))
        self.lora_B = nn.Parameter(torch.empty((self.r, output_dim)))
        # self.lora_A = nn.Linear(input_dim, self.r, bias=False)
        # self.lora_B = nn.Linear(self.r, output_dim, bias=False)
        self.scaling = self.lora_alpha / self.r

    def init(self, device, dtype):
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            state_dict[name] = torch.zeros_like(param, device=device, dtype=dtype)
        self.load_state_dict(state_dict=state_dict, assign=True)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.lora_dropout(x @ self.lora_A @ self.lora_B)

class LoRAWrapper(nn.Module):
    def __init__(self, 
                 module:nn.Module, 
                 input_dim:int, 
                 output_dim:int, 
                 r:int=8, 
                 lora_alpha:int=8, 
                 lora_dropout:float=0.1, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_module = module
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.lora = LoRA(input_dim=input_dim, output_dim=output_dim, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def init_lora(self):
        submodule = next(self.lora_module.parameters())
        self.lora.init(device=submodule.device, dtype=submodule.dtype)
        for p in self.lora_module.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return self.lora_module(x) + self.lora(x)

def wrap_linear(module:nn.Module, 
                r:int=16, 
                lora_alpha:int=32, 
                lora_dropout:float=0.05, 
                lora_num:int=1) -> nn.Module:
    def wrap_lora(m):
        for _ in range(lora_num):
            m = LoRAWrapper(m, 
                           m.in_features, 
                           m.out_features, 
                           r=r, 
                           lora_alpha=lora_alpha, 
                           lora_dropout=lora_dropout)
        return m
    if isinstance(module, nn.Linear):
        return wrap_lora(module)
    for name, submodule in list(module.named_modules()):
        if isinstance(submodule, nn.Linear):
            attributes = name.split('.')
            current_obj = module
            for attr in attributes[:-1]:
                current_obj = getattr(current_obj, attr)
            new_submodule = wrap_lora(submodule)
            setattr(current_obj, attributes[-1], new_submodule)
    return module

def frozen_parameter_not_in_lora(model:nn.Module):
    for name, param in model.parameters():
        param.requires_grad = ('lora' in name and 'lora_module' not in name)
        