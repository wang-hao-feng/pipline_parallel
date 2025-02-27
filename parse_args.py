import os
import torch
import argparse
from models import model_regisiter
from prompt import prompt_regisiter
from loss_fn import loss_fn_regisiter, sim_fn_regisiter
from dataset import dataset_regisiter
from prompt_fn import prompt_fn_regisiter, image_token_prompt_regisiter

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型
    parser.add_argument('-m', '--model', type=str, choices=model_regisiter.keys())
    parser.add_argument('-mp', '--model-path', type=str)
    parser.add_argument('-cp', '--config-path', type=str)
    parser.add_argument('-pp', '--processor-path', type=str)

    parser.add_argument('--frozen_vit', action='store_true')
    parser.add_argument('--frozen_mlp', action='store_true')
    parser.add_argument('--frozen_llm', action='store_true')
    parser.add_argument('--frozen_lm_head', action='store_true')

    parser.add_argument('--vit_lora_num', type=int)
    parser.add_argument('--mlp_lora_num', type=int)
    parser.add_argument('--llm_lora_num', type=int)

    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--nf4', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--low_cpu_mem_usage', action='store_true')
    parser.add_argument('--use_flash_attention_2', action='store_true')

    # 数据集
    parser.add_argument('-d', '--dataset', type=str, choices=dataset_regisiter.keys())
    parser.add_argument('-dp', '--dataset-path', type=str)
    parser.add_argument('--vsr_split_type', type=str, choices=['random', 'zeroshot'])

    parser.add_argument('-bs', '--batch-size', type=int)
    parser.add_argument('-mbs', '--micro-batch-size', type=int)

    # prompt
    parser.add_argument('-tp', '--text_prompt', type=str, choices=prompt_regisiter.keys())
    parser.add_argument('-pf', '--prompt_fn', type=str, choices=prompt_fn_regisiter.keys())
    parser.add_argument('-itp', '--image_token_prompt', type=str, choices=image_token_prompt_regisiter.keys())

    parser.add_argument('-r', '--result_path', type=str, default='./results')
    parser.add_argument('-o', '--output', type=str, default='result.json')

    # train
    parser.add_argument('-lr', '--learning_rate', type=float, default=4e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05)

    parser.add_argument('-ml', '--min_lr', type=float, default=2e-5)
    parser.add_argument('-wusl', '--warm_up_start_lr', type=float, default=3e-5)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=100)
    parser.add_argument('-es', '--eval_steps', type=int, default=200)
    parser.add_argument('-ss', '--save_steps', type=int, default=200)
    parser.add_argument('-ts', '--total_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--max_token_num', type=int)
    parser.add_argument('-t', '--task', type=str, choices=loss_fn_regisiter.keys())
    parser.add_argument('--sim_fn', type=str, choices=sim_fn_regisiter.keys())
    parser.add_argument('-nn', '--negative-num', type=int)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    model_kwargs = {}
    if args.bf16:
        model_kwargs['torch_dtype'] = torch.bfloat16
    if args.fp16:
        model_kwargs['torch_dtype'] = torch.float16
    if args.nf4:
        model_kwargs['load_in_4bit'] = True
        model_kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if args.load_in_8bit:
        model_kwargs['load_in_8bit'] = True
    if args.low_cpu_mem_usage:
        model_kwargs['low_cpu_mem_usage'] = True
    if args.use_flash_attention_2:
        model_kwargs['use_flash_attention_2'] = True
    if args.config_path is not None:
        model_kwargs['config_path'] = os.path.expanduser(args.config_path)
    if args.processor_path is not None:
        model_kwargs['processor_path'] = os.path.expanduser(args.processor_path)
    if args.frozen_vit:
        model_kwargs['frozen_vit'] = args.frozen_vit
    if args.frozen_mlp:
        model_kwargs['frozen_mlp'] = args.frozen_mlp
    if args.frozen_llm:
        model_kwargs['frozen_llm'] = args.frozen_llm
    if args.frozen_lm_head:
        model_kwargs['frozen_lm_head'] = args.frozen_lm_head
    if args.vit_lora_num is not None:
        model_kwargs['vit_lora_num'] = args.vit_lora_num
    if args.mlp_lora_num is not None:
        model_kwargs['mlp_lora_num'] = args.mlp_lora_num
    if args.llm_lora_num is not None:
        model_kwargs['llm_lora_num'] = args.llm_lora_num

    dataset_kwargs = {}
    if args.vsr_split_type:
        dataset_kwargs['split_type'] = args.vsr_split_type
    if args.batch_size:
        dataset_kwargs['batch_size'] = args.batch_size
        if args.micro_batch_size:
            dataset_kwargs['micro_batch_size'] = args.micro_batch_size
    
    loss_fn_kwargs = {}
    if args.negative_num:
        loss_fn_kwargs['negative_num'] = args.negative_num
    if args.sim_fn:
        loss_fn_kwargs['sim_fn'] = args.sim_fn

    return args, model_kwargs, dataset_kwargs, loss_fn_kwargs