import math
import torch

class ModelCutter():
    def __init__(self):
        self.intervl_num_layers = {
            '1B': 24, '2B': 24, '4B': 32, '8B': 32,
            '26B': 48, '40B': 60, '76B': 80}
        self.world_size = torch.cuda.device_count()

    def cut_layer(self, num_layers:int, vision_gpu_ratio:float):
        num_layers_per_gpu = math.ceil(num_layers / (self.world_size - vision_gpu_ratio))
        num_layers_per_gpu = [num_layers_per_gpu] * self.world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vision_gpu_ratio))
        return num_layers_per_gpu

    def cut_language_model(self, num_layers:int, visual_gup_ratio:float, train:bool=False):
        device_map = {}
        layer_cnt = 0
        num_layers_per_gpu = self.cut_layer(num_layers, visual_gup_ratio)
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0 if not train else self.world_size-1
        if not train:
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    def split_intervl_model(self, size:str, train:bool=False):
        num_layers = self.intervl_num_layers[size.upper()]
        device_map = self.cut_language_model(num_layers, 0.5, train=train)
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.output'] = 0 if not train else self.world_size-1
        return device_map
    
    def split_llava_next(self, train:bool=False):
        num_layers = 60
        device_map = self.cut_language_model(num_layers, 0.5, train=train)
        device_map['image_newline'] = 0
        device_map['vision_tower'] = 0
        device_map['multi_modal_projector'] = 0
        return device_map

    def split_llama_3_2(self, train:bool=False):
        num_layers = 100
        device_map = self.cut_language_model(num_layers, 0.1, train=train)
        device_map['vision_model'] = 0
        device_map['multi_modal_projector'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        return device_map

cutter = ModelCutter()