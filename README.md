# 流水并行
## 注册表
本框架内所有可变内容均使用注册表进行管理，regisiter.py实现了注册表的基本功能。
注册表可以视为一个字典，能按正常字典的方式使用（如prompt.py内保存的prompt），也可以通过装饰器的形式，将类或函数添加进注册表内（如loss_fn.py）。通过在sh脚本内替换对应的key，可以实现代码内模块的替换。

## 模型
### 模型切分
后续出现的visual_gup_ratio参数或者变量用于定义视觉模型需要占多大的显卡（大于0的小数）。对于纯语言模型，设置为0即可。huggingface中常用的CausalModel通常有如下格式：\
CausalModel\
├── LanguageModel\
│&emsp;&emsp;├── tok_embeddings/embed_tokens\
│&emsp;&emsp;├── layers\
│&emsp;&emsp;└── norm\
└── output/lm_head

#### 推理时切分
由于推理时需要将输出返回给输入，存在环，因此并未实现推理时的流水并行，只实现了多卡推理。ModelCutter.py中的cuttter实现了对huggingface模型的切分，并返回一个device_map。device_map为一个字典，key为模型内模块的名字，value为key所在的gpu号。拆分模型时需要注意以下两点：
1. 只需要拆分layers。
2. layers的最后一层要和第一层在同一张显卡上。
#### 流水并行切分
在模型参数的切分上，流水并行与推理时的切分差别不大，不同点在于最后一层和第一次不需要在同一张显卡上。但这仅仅是逻辑上相似，在具体实现上有很大不同。流水并行的模型需要重写部分函数，可以参考pp_models.py中的InternVL2流水的实现。
1. 重写__init__：
    * 参数说明：
        * config：模型的配置文件，受显存和内存大小限制，我们不可能直接使用from_pretrained函数加载多份模型参数，因此需要将模型构造和参数加载分开。config参数则包含了构造模型的所有信息，配置文件与模型参数文件在同一问价内，只需要在执行脚本时设置config-path即可自动加载。
        * stage：表示这一块属于整个模型的第几块，stage将定义后续forward的行为。
        * world_size：在单机流水并行中，world_size即表示有几张显卡，也表示模型被拆分成了几块。
        * 其余参数可根据实际需求设置，并在parse_args.py中将其添加进model_kwargs内。
    * 需要实现的功能:
        * 通过config加载模型包含原始模型所有的模块，因此需要根据实际情况删除不需要的模块，减少内存和显存占用。
            * 由于layers的模块名称包含当前层的序号，删除多余层时，尽可能使用某些类(None, nn.Identity)占住被删除layer的位置，从而保证模块名称不变。
        * 如果使用lora微调，要在需要的模块上添加lora，具体实现请参考[lora](#lora)。
2. 重写forward函数
    * 参数说明：
        * 模型不同部分的输入可能不同，同时，流水并行的通信只能传输Tensor无法传输dict，因此建议使用*args作为参数。输入函数中的所有未命名的参数将被按顺序存入args和kwargs中，kwargs是一个字典，存储了所有以key-value方式存储的参数；args是一个列表，按顺序存储了剩下的所有未命名参数。利用python的传参机制，可以实现任意数量的可变参数传递，适用于此处(如多模态模型，stage0的输入包含图片，后续的输入则不需要)。
    * 需要实现的功能：
        * 参数解析
        * 删除不需要层后的froward逻辑
            * 使用stage和world_size判断当前是那一块。
            * 可以复制源代码，删除不需要的部分来实现。

3. 实现输入样例函数(get_example_input)
    通信时需要确定传输内容的大小，因此pytorch要求必须给出输入样例。输出可以通过执行forward实现，因此可以不实现。
    * 参数说明
        * micro_batch_size：流水线时每个小batch的大小。
        * (可选)max_token_num：上下文长度，如果不设置，会默认使用config中设置的上下文长度，通常会很大，会导致显存占用过多且降低训练速度，因此建议根据数据进行设置。
    * 需要实现的功能：
        * 当前层所需的所有输入的样例(可以用torch.zero构造)
            * 数据类型必须与上层输出的数据类型相同，使用半精度训练时，层间输入均为torch.float16或torch.bfloat16，此时如果样例为torch.float32类型，则会出现不可知的错误

4. lora名称映射(get_key_map_without_last_lora)
    * 如果使用lora微调，模型中参数的名称可能会改变，因此需要提供一个映射供加载参数时使用。映射为一个字典，key为原参数名称，value为添加lora后的参数名称
    * 如果没有使用lora，返回的字典中key==value即可

5. frozen函数
    * 将所有不需要训练的参数的require_grad设置为False，需要训练的设为True

### lora
* modules/lora.py中实现了一个lora包装类LoRAWrapper，可以为任意一个module添加上lora，但会改变参数的名称，例如：\
module\
└── linear\
 &emsp;&emsp;└── weight\
被LoRAWrapper包装后变为\
module\
└── linear\
&emsp;&emsp;├── lora_module\
&emsp;&emsp;│&emsp;&emsp;└── weight\
&emsp;&emsp;└── lora\
&emsp;&emsp;&emsp;&emsp;├── lora_A.weight\
&emsp;&emsp;&emsp;&emsp;└── lora_B.weight
* wrap_linear函数能够为module内的所有Linear层添加lora，并返回添加lora后的module。

### 模型加载
#### 推理
加载推理模型的函数均保存在load_model.py内，并通过builder_regisiter进行管理。加载函数输入模型存储的路径，输出模型、tokenizer\processor、device_map三个参数。加载模型所需的其他参数请再parse_args.py中设置，并添加到model_kwargs中，evaluate.py会通过**kwargs将其送入加载函数中。
#### 流水并行
加载流水并行模型的函数保存在pp_models.py内，并通过train_model_regisiter进行管理。
* 参数说明
    * 输入
        * config_path：配置文件所在文件夹的路径
        * processor_path：tokenizer或processor所在文件夹的路径
        * checkpoint_path：模型参数所在文件夹的路径
        * stage：见[流水并行切分](#模型切分)
        * world_size：见[流水并行切分](#模型切分)
    * 输出
        * model：加载的模型
        * tokenizer或processor：加载的tokenizer或processor
        * pad_token_id：用于计算loss时忽略pad_token
在加载模型时，建议使用with accelerate.init_empty_weights()包装构造模型过程，使得加载的参数在meta设备中，从而不占用显存。pp_utils.py中实现了一些额外的工具，load_splited_model函数能加载分块model的参数，因此具体加载参数的内容不需要在函数内实现，只需要调用load_spited_model即可。

## 推理模型
models.py中还保存了所有推理的函数，并通过model_regisiter管理。这里的函数需要实现处理输入(如处理图像、tokenize)，并在此处调用chat或generate生成内容并返回。

## prompt
为了解耦不同模型的prompt设计，prompt被分为模板和模板处理函数两部分。
### 模板
所有prompt模板均保存在prompt.py中，并使用prompt_regisiter进行管理。
### 文本处理函数
文本处理函数保存在prompt_fn.py中，并使用prompt_fn_regisiter进行管理。
* 参数说明
    * text：数据集给出的字典(第二个返回值)
    * prompt：模板
    * 返回处理好的prompt
### 图像处理函数
图像处理函数也保存在prompt_fn.py中，使用image_token_prompt_regisiter管理。图像处理函数用于将prompt模板中图像的位置替换成模型所需的图像token(通常为\<image\>或没有)。
* 参数说明
    * prompt：模板
    * images：图像列表

## 数据
### 数据集
dataset.py中需要实现数据集加载，并通过dataset_regisiter管理。额外的参数需要在parse_ags.py中实现并添加到dataset_kwargs中。第一个返回值为image，纯文本训练可以返回None；第二个返回值为一个字典，用于后续prompt函数处理。在加载数据时，为了降低内存占用，最好只在流水线开头和结尾返回数据。
### collate_fn
同样在dataset.py中实现，通过collate_fn_regisiter管理。与数据集相同，为了降低内存占用，最好只在流水线开头和结尾返回数据。
* 参数说明
    * batchs: DataLoader载入的batch
    * prompt: prompt模板
    * max_token_num: 上下文长度
    * prompt_fn: 处理文本的函数
    * image_token_prompt_fn: 替换图像token的函数
    * processor: tokenizer或processor，用于处理输入
    * model_name: 保留此参数或换成**kwargs即可
    * rank: 同stage
    * world_size: 见[流水并行切分](#模型切分)

## loss函数
loss函数均保存在loss_fn中，通过loss_fn_regisiter管理。