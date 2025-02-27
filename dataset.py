import os
import re
import json
from PIL import Image
from typing import Literal, Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import process_image
from regisiter import Regisiter

dataset_regisiter = Regisiter()
collate_fn_regisiter = Regisiter()

@dataset_regisiter('vsr')
class VSR(Dataset):
    def __init__(self, 
                 path:str, 
                 split_type:Literal['random', 'zeroshot']='random', 
                 split:Literal['train', 'dev', 'test']='train', 
                 **kwargs):
        super().__init__()
        self.path = os.path.expanduser(path)
        with open(os.path.join(self.path, f'vsr_{split_type}/{split}.jsonl'), encoding='utf-8') as f:
            self.datas = [json.loads(line) for line in f.readlines()]
    
    def __getitem__(self, index):
        data = self.datas[index]
        image = Image.open(os.path.join(self.path, 'images/{0}'.format(data['image'])))
        caption = data['caption']
        answer = ['false', 'true'][data['label']]
        return image, {'caption': caption, 'answer':answer, 'relation': data['relation']}
    
    def __len__(self):
        return len(self.datas)

@dataset_regisiter('spatial_mm')
class SpatialMM(Dataset):
    def __init__(self, path:str, **kwargs):
        super().__init__()
        self.path = os.path.expanduser(path)
        self.datas = []
        for number in ['one', 'two']:
            with open(os.path.join(self.path, f'spatial_mm_{number}_obj.json'), encoding='utf-8') as f:
                self.datas += json.load(f)
        self.datas = [data for data in self.datas if os.path.exists(os.path.join(self.path, 'Spatial_MM_Obj/{}'.format(data['image_name'])))]
    
    def __getitem__(self, index):
        data = self.datas[index]
        image = Image.open(os.path.join(self.path, 'Spatial_MM_Obj/{}'.format(data['image_name'])))
        return image, data
    
    def __len__(self):
        return len(self.datas)

@dataset_regisiter('whats_up')
class WhatsUp(Dataset):
    def __init__(self, path:str, **kwargs):
        super().__init__()
        self.path = os.path.expanduser(path)
        self.datas = []
        for subset in ['clevr', 'images']:
            with open(os.path.join(self.path, f'controlled_{subset}_dataset.json')) as f:
                self.datas += json.load(f)
    
    def __getitem__(self, index):
        data = self.datas[index]
        image = Image.open(os.path.join(self.path, data['image_path']))
        return image, data

    def __len__(self):
        return len(self.datas)

@dataset_regisiter('rotate-caption')
class ROTATE_CAPTION(Dataset):
    def __init__(self, 
                 path:str, 
                 split:Literal['train', 'val'], 
                 rank:int, 
                 world_size:int, 
                 **kwargs):
        super().__init__()
        self.image_path = os.path.join(path, 'images')
        self.rank = rank
        self.world_size = world_size
        with open(os.path.join(path, f'captions_{split}.jsonl'), encoding='utf-8') as f:
            self.captions = [json.loads(line) if rank == 0 or rank == world_size - 1 else {} for line in f.readlines()]
    
    def __getitem__(self, index):
        if self.rank == 0 or self.rank == self.world_size - 1:
            data = self.captions[index]
            images_name = [data['image']] + data['positive_images'] + data['negative_images']
            images = [Image.open(os.path.join(self.image_path, name)) for name in images_name]
            captions = [data['caption']] + [data['positive_caption_map'][name] for name in data['positive_images']] + [data     ['negative_caption_map'][name] for name in data['negative_images']]
            return images, [{'caption': caption} for caption in captions]
        else:
            return [], []
    
    def __len__(self):
        return len(self.captions)

@dataset_regisiter('rotate-qa')
class ROTATE_QA(Dataset):
    def __init__(self, 
                 path:str, 
                 split:Literal['train', 'val', 'test'], 
                 rank:int=0, 
                 world_size:int=1, 
                 **kwargs):
        super().__init__()
        self.image_path = os.path.join(path, 'images')
        self.rank = rank
        self.world_size = world_size
        with open(os.path.join(path, f'qas_{split}.jsonl'), encoding='utf-8') as f:
            self.qas = [json.loads(line)  if rank == 0 or rank == world_size - 1 else {} for line in f.readlines()]
    
    def __getitem__(self, index):
        if self.rank == 0 or self.rank == self.world_size - 1:
            qa = self.qas[index]
            image = Image.open(os.path.join(self.image_path, qa['image']))
            return image, qa
        else:
            return [], []
    
    def __len__(self):
        return len(self.qas)

@collate_fn_regisiter('InternVL2')
def internvl_collate_fn(batchs:list, 
                        prompt:str, 
                        max_token_num:int, 
                        prompt_fn:Callable, 
                        image_token_prompt_fn:Callable, 
                        processor:PreTrainedTokenizer, 
                        rank:int, 
                        world_size:int):
    images = [batch[0] for batch in batchs]
    texts = [batch[1] for batch in batchs]
    IMG_START_TOKEN = '<img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    num_image_token = 256
    IMG_END_TOKEN = '</img>'
    if rank == 0 or rank == world_size - 1:
        pixel_values = [process_image(image) for image in images]
        querys = [prompt_fn(text, prompt) for text in texts]
        querys = [image_token_prompt_fn(query, image) for image, query in zip(images, querys)]
        num_patches_lists = [[pixel_value.shape[0]] for pixel_value in pixel_values]
        for i, query in enumerate(querys):
            num_patches_list = num_patches_lists[i]
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            querys[i] = query
        processor.padding_side = 'right'
        inputs = processor(querys, return_tensors='pt', max_length=max_token_num, padding='max_length')
        pixel_values = torch.concat(pixel_values)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        image_flags = torch.ones((input_ids.shape[0], ))
        # 获取label
        instructs = [query.split('<|im_start|>assistant\n')[0] + '<|im_start|>assistant\n' for query in querys]
        instructs_mask = processor(instructs, return_tensors='pt', max_length=max_token_num, padding='max_length')['input_ids'] != processor.pad_token_id
        labels = instructs_mask * processor.pad_token_id + input_ids * torch.logical_not(instructs_mask)
        # instructs_len = [processor(instruct, return_tensors='pt', max_length=max_token_num)['input_ids'].shape[1] for instruct in instructs]
        # labels = input_ids.clone()
        # for i, length in enumerate(instructs_len):
        #     labels[i, :length] = processor.pad_token_id
        return (pixel_values, input_ids, image_flags, attention_mask) if rank == 0 else tuple(), labels
    return None, None

answers = sorted(['front', 'back', 'left', 'right', 'front-left', 'front-right', 'back-left', 'back-right', 
                   '1 o\'clock', '2 o\'clock', '3 o\'clock', '4 o\'clock', '5 o\'clock', '6 o\'clock', 
                   '7 o\'clock', '8 o\'clock', '9 o\'clock', '10 o\'clock', '11 o\'clock', '12 o\'clock', 
                   'one o\'clock', 'two o\'clock', 'three o\'clock', 'four o\'clock', 'five o\'clock', 'six o\'clock', 
                   'seven o\'clock', 'eight o\'clock', 'nine o\'clock', 'ten o\'clock', 'eleventh o\'clock', 'twelve o\'clock'], key=lambda x:-len(x))
pattern = '|'.join([re.escape(answer) for answer in answers])
regex = re.compile(pattern)

@collate_fn_regisiter('contrastive')
def contrastive_collate_fn(batchs:list, 
                           prompt:str, 
                           max_token_num:int, 
                           prompt_fn:Callable, 
                           image_token_prompt_fn:Callable, 
                           processor:PreTrainedTokenizer, 
                           model_name:str, 
                           rank:int, 
                           world_size:int):
    model_collate_fn = collate_fn_regisiter[model_name]
    images = sum([batch[0] for batch in batchs], start=[])
    texts = sum([batch[1] for batch in batchs], start=[])
    inputs, labels = model_collate_fn(batchs=[[image, text] for image, text in zip(images, texts)], 
                                      prompt=prompt, 
                                      max_token_num=max_token_num, 
                                      prompt_fn=prompt_fn, 
                                      image_token_prompt_fn=image_token_prompt_fn, 
                                      processor=processor, 
                                      rank=rank, 
                                      world_size=world_size)
    if labels is not None:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        querys = processor.batch_decode(labels, skip_special_tokens=False)
        for i, query in enumerate(querys):
            prefix_pairs = [(query[:match.start()], query[:match.end()]) for match in regex.finditer(query)]
            idx_pairs = [(len(processor(prefix1)['input_ids'])-2, len(processor(prefix2)['input_ids'])-2) for prefix1, prefix2 in prefix_pairs]
            for start_idx, end_idx in idx_pairs:
                mask[i, start_idx:(end_idx + (end_idx < max_token_num))] = True
        labels[mask] *= -1
    return inputs, labels

@collate_fn_regisiter('sft')
def sft_collate_fn(batchs:list, 
                   prompt:str, 
                   max_token_num:int, 
                   prompt_fn:Callable, 
                   image_token_prompt_fn:Callable, 
                   processor:PreTrainedTokenizer, 
                   model_name:str, 
                   rank:int, 
                   world_size:int):
    model_collate_fn = collate_fn_regisiter[model_name]
    images = [batch[0] for batch in batchs]
    texts = [batch[1] for batch in batchs]
    if isinstance(images[0], list):
        images = sum(images, start=[])
    if isinstance(texts[0], list):
        texts = sum(texts, start=[])
    return model_collate_fn(batchs=[[image, text] for image, text in zip(images, texts)], 
                            prompt=prompt, 
                            max_token_num=max_token_num, 
                            prompt_fn=prompt_fn, 
                            image_token_prompt_fn=image_token_prompt_fn, 
                            processor=processor, 
                            rank=rank, 
                            world_size=world_size)

if __name__ == '__main__':
    dataset = WhatsUp('~/datasets/WhatsUp')
    num = 0
    for data in dataset.datas:
        num += (not os.path.exists(os.path.join(dataset.path, data['image_path'])))
    print(num)
    print(len(dataset))