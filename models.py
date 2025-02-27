import os
from PIL import Image

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import MllamaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from utils import *
from regisiter import Regisiter
from load_model import builder_regisiter

model_regisiter = Regisiter()

@model_regisiter('GPT-4o')
def GPT(text:str, 
        images:Image.Image|list[Image.Image], 
        **kwargs):
    images = base64_images(images)
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        }, 
        {
            'role': 'user', 
            'content': [
                {
                    'type': 'text', 
                    'text': text, 
                }
            ]
        }
    ]
    response = {"custom_id": '', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": messages, "max_tokens": 128}}
    return response

@model_regisiter('InternVL2-4B')
@model_regisiter('InternVL2-8B')
@model_regisiter('InternVL2-26B')
@model_regisiter('InternVL2-40B')
@model_regisiter('InternVL2-Llama3-76B')
@model_regisiter('InternVL2-8B-rotate')
def InternVL(text:str, 
             images:Image.Image|list[Image.Image], 
             model:PreTrainedModel, 
             processor:PreTrainedTokenizer):
    images = [process_image(image).to(dtype=model.dtype).cuda() for image in images]
    image = torch.cat(images, dim=0) if len(images) > 0 else None
    prompt = text
    generation_config = dict(max_new_tokens=64, do_sample=False)
    output = model.chat(processor, image, prompt, generation_config)
    return output

@model_regisiter('Llava-Next-34B')
def LlavaNext34B(text:str, 
                 images:Image.Image|list[Image.Image], 
                 model:LlavaNextForConditionalGeneration, 
                 processor:LlavaNextProcessor):
    images = padding_images(images) if len(images) > 0 else None
    prompt = f'<|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n'
    inputs = processor(prompt, images, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    return processor.decode(output[0][2:], skip_special_tokens=True).split('assistant')[-1]

@model_regisiter('Llama-3.2-11B')
@model_regisiter('Llama-3.2-90B')
def Llama_3_2(text:str, 
              images:Image.Image|list[Image.Image], 
              model:MllamaForConditionalGeneration, 
              processor:PreTrainedTokenizer):
    messages = [
    {
        "role": "user", 
        "content": [{"type": "image"} for _ in images] + 
        [{"type": "text", "text": text}]
    }]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

class Model():
    def __init__(self, 
                 model:str, 
                 path:str, 
                 **kwargs):
        self.model, self.processor, self.device_map = builder_regisiter[model](os.path.expanduser(path), low_cpu_mem_usage=True, **kwargs)
        self.model_func = model_regisiter[model]
    
    def __call__(self, text:str, images:Image.Image|list[Image.Image]):
        if isinstance(images, Image.Image):
            images = [images]
        return self.model_func(text=text, images=images, model=self.model, processor=self.processor)
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device