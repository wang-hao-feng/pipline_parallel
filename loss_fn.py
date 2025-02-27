import torch
import torch.nn as nn
import torch.nn.functional as F

from regisiter import Regisiter

loss_fn_regisiter = Regisiter()
sim_fn_regisiter = Regisiter()

contrastive_call_times = 0

@loss_fn_regisiter('sft')
def next_token_loss_fn(outputs:torch.Tensor|tuple,  
                       targets:torch.Tensor, 
                       ignore_token_id:int, 
                       **kwargs):
    
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]).contiguous()
    shift_targets = targets[..., 1:].reshape(-1).contiguous()
    ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_token_id)
    loss = ce_fn(shift_logits, shift_targets)
    return loss

@sim_fn_regisiter('js')
def js_div(logits:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, seq_len, vocab_size)
    mask: (batch_size, seq_len, 1)
    output: (batch_size - 1, )
    """
    kl_fn = nn.KLDivLoss(reduction='none', log_target=True)
    mask = (torch.sum(mask, dim=0) != 0).squeeze(-1)
    logits = logits[:, mask, :]
    seq_len = logits.shape[1]
    # log_logits = F.softmax(logits, dim=-1)
    # log_logits = torch.log(torch.sum(log_logits * mask, dim=1) / torch.sum(mask, dim=1))
    log_logits = F.log_softmax(logits, dim=-1)
    avg_logits = (log_logits[0] + log_logits[1:]) * 0.5
    kl1 = torch.sum(kl_fn(log_logits[0], avg_logits), dim=tuple(range(1, len(log_logits.shape))))
    kl2 = torch.sum(kl_fn(log_logits[1:], avg_logits), dim=tuple(range(1, len(log_logits.shape))))
    js = 0.5 * (kl1 + kl2) / seq_len
    return js

@sim_fn_regisiter('cos')
def cos_sim(logits:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, seq_len, vocab_size)
    mask: (batch_size, seq_len, 1)
    output: (batch_size - 1, )
    """
    predict = F.softmax(logits, dim=-1)
    predict = torch.sum(predict * mask, dim=1) / torch.sum(mask, dim=1)
    sim = (predict[1:] @ predict[:1].T).view(-1)
    return sim

@loss_fn_regisiter('contrastive')
def contrastive_loss_fn(outputs:torch.Tensor, 
                        targets:torch.Tensor, 
                        ignore_token_id:int, 
                        negative_num:int, 
                        n_microbatches:int, 
                        total_steps:int, 
                        warm_up_steps:int=0, 
                        update:bool=True, 
                        sim_fn:str='js'):
    """
    outputs: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    global contrastive_call_times
    step = int(contrastive_call_times // n_microbatches)
    if update:
        contrastive_call_times += 1
    if isinstance(outputs, tuple):
        outputs = outputs[0] if len(outputs[0].shape) > 0 else outputs[1]
    # 全局 next token predict部分
    global_next_token_loss = next_token_loss_fn(outputs, torch.abs(targets), ignore_token_id)
    # if step < warm_up_steps:
    #     return global_next_token_loss
    local_mask = (targets < 0) if sim_fn != 'js' else (targets != ignore_token_id)
    local_targets = targets.clone()
    local_targets[local_targets > 0] = ignore_token_id
    local_next_token_loss = next_token_loss_fn(outputs, torch.abs(local_targets), ignore_token_id)
    # 对比学习部分
    batch_size, seq_len, vocab_size = outputs.shape
    group_size = negative_num + 2
    assert batch_size % group_size == 0
    groups = outputs.view(-1, group_size, seq_len, vocab_size).contiguous()
    groups_mask = local_mask.view(-1, group_size, seq_len).unsqueeze(-1).contiguous()
    group_num = groups.shape[0]
    group_sims = torch.stack([sim_fn_regisiter[sim_fn](group, mask) for group, mask in zip(groups, groups_mask)])    # (group_num, group_size - 1)
    group_sims = F.softmax(group_sims, dim=-1)
    ce_fn = nn.CrossEntropyLoss()
    contrastive_loss = ce_fn(group_sims, torch.zeros((group_num,), device=group_sims.device, dtype=torch.int64))
    
    # alpha = -math.cos((step - warm_up_steps) / (total_steps - warm_up_steps) * math.pi) * 0.5 + 0.5 
    # alpha = (step - warm_up_steps) / (total_steps - warm_up_steps)
    return global_next_token_loss + local_next_token_loss + contrastive_loss
    # return global_next_token_loss + alpha * contrastive_loss