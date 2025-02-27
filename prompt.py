from regisiter import Regisiter

prompt_regisiter = Regisiter()

VSR_ZEROSHOT = """\
[IMAGE]Based on the image, is this statement true or false? "[CAPTION]" Answer:
"""
prompt_regisiter['vsr_zeroshot'] = VSR_ZEROSHOT

SPATIAL_MM = """\
[IMAGE]Please select the correct option.\n [QUESTION]\n Your option:
"""
prompt_regisiter['spatial_mm'] = SPATIAL_MM

WHATS_UP = """\
[IMAGE]Please choose the caption that best matches the picture from the perspective of the camera itself.\n [CAPTIONS]\n Your option:
"""
prompt_regisiter['whats_up'] = WHATS_UP

INTERNVL2_ROTATE_QA = """\
<|im_start|>system
你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n
[IMAGE][QUESTION]<|im_end|><|im_start|>assistant\n[ANSWER]<|im_end|>
"""
prompt_regisiter['internvl2_rotate_qa'] = INTERNVL2_ROTATE_QA

INTERNVL2_ROTATE_CAPTION = """\
<|im_start|>system
你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n
[IMAGE]Provide a detailed description of the spatial relationships between the objects in the image as much as possible.<|im_end|><|im_start|>assistant\n[CAPTION]<|im_end|>
"""
prompt_regisiter['internvl2_rotate_caption'] = INTERNVL2_ROTATE_CAPTION

ROTATE_QA = """\
[IMAGE]
I will ask you a question. If the question requires a yes/no judgment, answer strictly "Yes" or "No". For short-answer questions: if the answer is a direction, use only one of these: front, back, left, right, front-left, back-left, front-right, back-right. If the answer is a clock position, use the format "[X] o'clock" where [X] is a number (1-12) or its English word (e.g., "2 o'clock" or "two o'clock"). Provide only the answer without explanations.
Question: 
[QUESTION]
Answer:
"""
prompt_regisiter['rotate_qa'] = ROTATE_QA

ROTATE_QA_AFTER_SFT = """\
[IMAGE]
[QUESTION]
"""
prompt_regisiter['rotate_qa_after_sft'] = ROTATE_QA_AFTER_SFT