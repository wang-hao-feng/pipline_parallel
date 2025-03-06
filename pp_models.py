import math

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer
from transformers import PreTrainedTokenizer, LlavaNextProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from modules.internvl2.modeling_internvl_chat import InternVLChatModel
from modules.internvl2.modeling_internlm2 import InternLM2ForCausalLM, InternLM2Model, _import_flash_attn

from modules.lora import wrap_linear
from regisiter import Regisiter
from pp_utils import load_splited_model

train_model_regisiter = Regisiter()

class SplitInternLM2Model(InternLM2Model):
    def __init__(self, 
                 config, 
                 stage:int, 
                 world_size:int, 
                 use_lora:bool=True, 
                 lora_num:int=1):
        super().__init__(config)
        self.stage = stage
        self.world_size = world_size
        if stage > 0:
            del self.tok_embeddings
        if stage != world_size - 1 and stage != -1:
            del self.norm
        vision_gpu_ratio = 0.5
        num_layers = len(self.layers)
        num_layers_per_gpu = math.ceil(num_layers / (self.world_size - vision_gpu_ratio))
        num_layers_per_gpu = [num_layers_per_gpu] * self.world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vision_gpu_ratio))
        layer_stage = [i for i in range(self.world_size) for _ in range(num_layers_per_gpu[i])]
        for i in range(num_layers):
            if layer_stage[i] != stage and stage != -1:
                self.layers[i] = nn.Identity()
            elif use_lora:
                self.layers[i] = wrap_linear(self.layers[i], lora_num=lora_num)
        self.training = True
    
    def forward(self, 
                *args, **kwargs):
        input_ids = None
        if self.stage == -1:
            return super().forward(*args, **kwargs)
        attention_mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else (args[1] if len(args) > 1 else None)
        output_attentions = kwargs['output_attentions'] if 'output_attentions' in kwargs else self.config.output_attentions
        use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else self.config.use_cache

        return_dict = kwargs['return_dict'] if 'return_dict' in kwargs else self.config.use_return_dict

        past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if self.config.attn_implementation == 'flash_attention_2':
            _import_flash_attn()
        
        if self.stage == 0:
            # input_ids = args[0]
            input_ids = None
            inputs_embeds = args[0]
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape[:2]
            elif inputs_embeds is not None:
                batch_size, seq_length = inputs_embeds.shape[:2]
            else:
                raise ValueError('You have to specify either input_ids or inputs_embeds')
            
            seq_length_with_past = seq_length
            
            if past_key_values is not None:
                seq_length_with_past = seq_length_with_past + past_key_values_length

            if inputs_embeds is None:
                inputs_embeds = self.tok_embeddings(input_ids)

            if self.config.attn_implementation == 'flash_attention_2':
                # 2d mask is passed through the layers
                attention_mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else None
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                    )
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            
            hidden_states = inputs_embeds
        else:
            hidden_states = args[0]
            seq_length = hidden_states.shape[1]
        args = args[1:]

        position_ids = kwargs['position_ids'] if 'position_ids' in kwargs else None
        if position_ids is None:
            device = input_ids.device if input_ids is not None else hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        for idx, decoder_layer in enumerate(self.layers):
            if isinstance(decoder_layer, nn.Identity):
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

        if self.stage != self.world_size - 1:
            return hidden_states, attention_mask
        
        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

class SplitInternLM2ForCasulLM(InternLM2ForCausalLM):
    def __init__(self, 
                 config, 
                 stage:int, 
                 world_size:int, 
                 use_lora:bool=True, 
                 lora_num:int=1):
        super().__init__(config)
        self.stage = stage
        self.world_size = world_size
        self.model = SplitInternLM2Model(config=self.model.config, stage=stage, world_size=world_size, use_lora=use_lora, lora_num=lora_num)
        if stage != self.world_size - 1 and stage != -1:
            del self.output
        # else:
        #     self.output = wrap_linear(self.output, lora_num=lora_num)
    
    def forward(self, *args, **kwargs):
        """
        input_embed, attention_mask, labels /
        input_embed, attention_mask /
        hidden_states, attention_mask, labels /
        hidden_states, attention_mask
        """
        if self.stage == -1:
            return super().forward(*args, **kwargs)
        outputs = self.model(*args[:3], **kwargs)
        labels = kwargs.pop('labels') if 'labels' in kwargs else (args[2] if len(args) > 2 else None)
        if self.stage != self.world_size - 1:
            outputs, attention_mask = outputs
            return outputs, attention_mask
        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return_dict = kwargs.pop('return_dict') if 'return_dict' in kwargs else None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output

class SplitInternVLChatModel(InternVLChatModel):
    def __init__(self, 
                 config, 
                 stage:int, 
                 world_size:int, 
                 vision_model=None, 
                 language_model=None, 
                 use_flash_attn=False, 
                 frozen_vit:bool=False, 
                 frozen_mlp:bool=False, 
                 frozen_llm:bool=False, 
                 frozen_lm_head:bool=False, 
                 vit_lora_num:int=0, 
                 mlp_lora_num:int=0, 
                 llm_lora_num:int=0, 
                 **kwargs):
        super().__init__(config, vision_model, language_model, use_flash_attn)
        self.stage = stage
        self.world_size = world_size
        if stage > 0:
            del self.vision_model
            del self.mlp1
        else:
            self.vision_model = wrap_linear(self.vision_model, lora_num=vit_lora_num)
            # self.mlp1 = wrap_linear(self.mlp1, lora_num=mlp_lora_num)
        self.frozen_vit = frozen_vit
        self.frozen_mlp = frozen_mlp
        self.frozen_llm = frozen_llm
        self.frozen_lm_head = frozen_lm_head
        self.language_model = SplitInternLM2ForCasulLM(self.language_model.config, stage, world_size, use_lora=llm_lora_num > 0, lora_num=llm_lora_num)
   
    def forward(self, *args, **kwargs):
        """
        pixel_values, input_ids, image_flags, attention_mask, labels / 
        pixel_values, input_ids, image_flags, attention_mask / 
        input_embed, attention_mask, labels /
        input_embed, attention_mask /
        hidden_states, attention_mask, labels /
        hidden_states, attention_mask /
        """
        if self.stage == -1:
            return super().forward(*args, **kwargs)
        if self.stage == 0:
            input_embeds = self.vision_chunk_forward(*args, **kwargs)
        else:
            if 'input_embeds' in kwargs:
                input_embeds = kwargs.pop('input_embeds')
            else:
                input_embeds = args[0]
        outputs = self.language_model(input_embeds, *(args[3:] if self.stage == 0 else args[1:]), **kwargs)
        labels = kwargs.pop('labels') if 'labels' in kwargs else (args[4] if len(args) > 4 else (args[2] if len(args) > 2 and len(args) < 4 else None))
        if self.stage != self.world_size - 1:
            outputs, attention_mask = outputs
            if labels is not None:
                return outputs.to(torch.float32), attention_mask.to(torch.float32), labels
            else:
                return outputs.to(torch.float32), attention_mask.to(torch.float32)
        logits = outputs.logits if not isinstance(outputs, tuple) else (outputs[0] if labels is None else outputs[1])
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return_dict = kwargs.pop('return_dict') if 'return_dict' in kwargs else None
        if not return_dict:
            output = (logits.to(torch.float32),)
            return (loss.to(torch.float32),) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def vision_chunk_forward(self, 
                             pixel_values, 
                             input_ids = None, 
                             image_flags = None, 
                             *args, 
                             **kwargs):
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        return input_embeds

    def get_example_input(self, 
                          micro_batch_size:int, 
                          max_token_num:int=None, 
                          require_labels:bool=False):
        max_token_num = max_token_num if max_token_num is not None else self.config.llm_config.max_position_embeddings
        labels = torch.zeros((micro_batch_size, max_token_num), dtype=torch.long)
        if self.stage == 0:
            pixel_values = torch.zeros((micro_batch_size, 3, self.config.force_image_size, self.config.force_image_size), dtype=torch.float32)
            input_ids = torch.zeros((micro_batch_size, max_token_num), dtype=torch.long)
            image_flags = torch.zeros((micro_batch_size,), dtype=torch.float32)
            attention_mask = torch.zeros((micro_batch_size, max_token_num), dtype=torch.int64)
            if require_labels:
                return pixel_values, input_ids, image_flags, attention_mask, labels
            else:
                return pixel_values, input_ids, image_flags, attention_mask
        attention_mask = torch.zeros((micro_batch_size, 1, max_token_num, max_token_num))
        hidden_states = torch.zeros((micro_batch_size, max_token_num, self.config.llm_config.hidden_size))
        if require_labels:
            return hidden_states, attention_mask, labels
        else:
            return hidden_states, attention_mask
    
    def get_example_output(self, 
                           micro_batch_size:int, 
                           max_token_num:int=None):
        if self.stage == self.world_size - 1:
            logits = torch.zeros((micro_batch_size, max_token_num, self.config.llm_config.vocab_size))
            return logits
        hidden_state = torch.zeros((micro_batch_size, max_token_num, self.config.llm_config.hidden_size))
        attention_mask = torch.zeros((micro_batch_size, 1, max_token_num, max_token_num))
        return hidden_state, attention_mask

    def get_key_map_without_last_lora(self):
        key_map_without_last_lora = {}
        for key in self.state_dict():
            if ('vision_model' in key and not self.frozen_vit) or ('language_model' in key and not self.frozen_llm):
                key_map_without_last_lora[key] = key.replace('.lora_module', '', 1)
            else:
                key_map_without_last_lora[key] = key
        return key_map_without_last_lora

    def frozen(self):
        if self.stage == 0:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False if self.frozen_vit else 'lora' in name and 'lora_module' not in name 
            for name, param in self.mlp1.named_parameters():
                # param.requires_grad = False if self.frozen_mlp else 'lora' in name and 'lora_module' not in name
                param.requires_grad = not self.frozen_mlp
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False if self.frozen_llm else 'lora' in name and 'lora_module' not in name
        if self.stage == self.world_size - 1:
            for param in self.language_model.output.parameters():
                param.requires_grad = not self.frozen_lm_head

@train_model_regisiter('InternVL2-2B')
@train_model_regisiter('InternVL2-8B')
@train_model_regisiter('InternVL2-26B')
@train_model_regisiter('InternVL2-40B')
def load_splited_intervl2(config_path:str, 
                          processor_path:str, 
                          checkpoint_path:str, 
                          stage:int, 
                          world_size:int, 
                          use_flash_attention_2:bool=True, 
                          **kwargs) -> tuple[SplitInternVLChatModel, PreTrainedTokenizer|LlavaNextProcessor, int]:
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    if not use_flash_attention_2:
        config.llm_config.attn_implementation = 'eager'
        config.vision_config.use_flash_attn = False
    tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    with init_empty_weights():
        model = SplitInternVLChatModel(config=config, stage=stage, world_size=world_size, **kwargs)
    train = kwargs['train'] if 'train' in kwargs else True
    load_splited_model(model, checkpoint_path, train=train)
    model.img_context_token_id = img_context_token_id
    return model, tokenizer, tokenizer.pad_token_id