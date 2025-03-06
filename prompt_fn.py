import string
from PIL import Image
from regisiter import Regisiter

prompt_fn_regisiter = Regisiter()
image_token_prompt_regisiter = Regisiter()

@prompt_fn_regisiter('vsr')
def VSR(text:dict, prompt:str):
    prompt = prompt.replace('[CAPTION]', text['caption'])
    return prompt

@prompt_fn_regisiter('spatial_mm')
def SpatialMM(text:dict, prompt:str):
    question = text['question']
    # if text['object2'] != "":
    #     question = question.lower()[0] + question[1:]
    #     question = 'From {0}\'s perspective, '.format(text['object1']) + question
    prompt = prompt.replace('[QUESTION]', question)
    return prompt

@prompt_fn_regisiter('whats_up')
def WhatsUP(text:dict, prompt:str):
    captions = text['caption_options']
    options = [f'({string.ascii_uppercase[i]}) {caption}' for i, caption in enumerate(captions)]
    prompt = prompt.replace('[CAPTIONS]', ' '.join(options))
    return prompt

@prompt_fn_regisiter('rotate_qa')
def ROTATE(text:dict, prompt:str):
    prompt = prompt.replace('[QUESTION]', text['question'])
    prompt = prompt.replace('[ANSWER]', text['answer'])
    return prompt

@prompt_fn_regisiter('rotate_caption')
def ROTATE(text:dict, prompt:str):
    prompt = prompt.replace('[CAPTION]', text['caption'])
    return prompt

@image_token_prompt_regisiter('special_image_token')
def SpecialImageToken(prompt:str, images:Image.Image|list[Image.Image]) -> str:
    return prompt.replace('[IMAGE]', '<image>' * (1 if isinstance(images, Image.Image) else len(images)))

@image_token_prompt_regisiter('empty')
def Empty(prompt:str, images:Image.Image|list[Image.Image]) -> str:
    return prompt.replace('[IMAGE]', '')