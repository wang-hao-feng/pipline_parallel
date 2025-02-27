import os
import json
from tqdm import tqdm

import torch

from models import Model
from parse_args import parse_args
from dataset import dataset_regisiter
from prompt import prompt_regisiter
from prompt_fn import prompt_fn_regisiter, image_token_prompt_regisiter
from calculate_score import calculator_regisiter

args, model_kwargs, dataset_kwargs, _ = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = dataset_regisiter[args.dataset](path=os.path.expanduser(args.dataset_path), split='test', **dataset_kwargs)

model = Model(model=args.model, path=os.path.expanduser(args.model_path), train=False, **model_kwargs)
state_dict = model.model.state_dict()

base_prompt = prompt_regisiter[args.text_prompt]
text_prompt = prompt_fn_regisiter[args.prompt_fn]
image_token_prompt = image_token_prompt_regisiter[args.image_token_prompt]
caculator = calculator_regisiter[args.dataset]

output_path = os.path.join(args.result_path, args.output)
def save(obj):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

results = []
count = 0
if os.path.exists(output_path):
    with open(output_path, encoding='utf-8') as f:
        results = json.load(f)
with torch.inference_mode():
    for i, (images, original_text) in tqdm(enumerate(dataset), total=len(dataset)):
        if i < len(results):
            continue
        text = text_prompt(original_text, base_prompt)
        text = image_token_prompt(text, images)
        response = model(text, images)
        count += 1
        if args.debug and count >= 20:
            exit()
        # print(response)
        # exit()
        results.append(response)
        save(results)

metrics = caculator(results, [text for _, text in dataset])
for m in metrics:
    if isinstance(metrics[m], float):
        print(f'{m}: {metrics[m]}')