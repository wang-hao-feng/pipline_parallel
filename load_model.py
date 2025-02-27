import torch

from ModelCutter import cutter
from regisiter import Regisiter

from accelerate import init_empty_weights
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from pp_models import load_splited_intervl2

builder_regisiter = Regisiter()

@builder_regisiter('GPT-4o')
def gpt(*args, **kwargs):
    return torch.nn.Linear(1, 1), None, None

@builder_regisiter('InternVL2-8B')
@builder_regisiter('InternVL2-26B')
@builder_regisiter('InternVL2-40B')
@builder_regisiter('InternVL2-Llama3-76B')
def load_internvl2(path:str,
                   train:bool=False, 
                   **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    size = path.split('-')[-1].upper()
    device_map = cutter.split_intervl_model(size, train=train)
    process = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    load_device_map = device_map if not train else {key: 'cpu' for key in device_map}
    if train:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, device_map=load_device_map, trust_remote_code=True, **kwargs).eval()
    return model, process, device_map

@builder_regisiter('InternVL2-8B-rotate')
def load_internvl2_with_lora(path:str, 
                             train:bool=False, 
                             **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    model, tokenizer, _ = load_splited_intervl2(checkpoint_path=path, 
                                                stage=-1, 
                                                world_size=1, 
                                                train=train,  
                                                **kwargs)
    return model, tokenizer, {}

@builder_regisiter('Llava-Next-34B')
def load_llava_next(path:str, 
                    train:bool=False, 
                    **kwargs) -> tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor, dict]:
    size = path.split('-')[-1].upper()
    device_map = cutter.split_llava_next(train=train) if size == '34B' else 'auto'
    process = LlavaNextProcessor.from_pretrained(path)
    model = LlavaNextForConditionalGeneration.from_pretrained(path, device_map=device_map, **kwargs).eval()
    return model, process, device_map

@builder_regisiter('Llama-3.2-11B')
@builder_regisiter('Llama-3.2-90B')
def load_llama_3_2(path:str, 
                   train:bool=False, 
                   **kwargs) -> tuple[MllamaForConditionalGeneration, AutoProcessor, dict]:
    size = path.split('-')[-1].upper()
    device_map = cutter.split_llama_3_2() if size == '90B' else 'auto'
    process = AutoProcessor.from_pretrained(path, train=train)
    model = MllamaForConditionalGeneration.from_pretrained(path, device_map=device_map, **kwargs).eval()
    return model, process, device_map