#!/bin/env zsh
#SBATCH -J internvl2-8b-w-contrastive
#SBATCH -o output/test/%x.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=25G
#SBATCH -t 24:00:00
#SBATCH -p compute 
#SBATCH --gres=gpu:a100-pcie-40gb:1

source ~/.zshrc

cd ~/rotate

conda activate torch25

NAME='internvl2-8b-w-contrastive'
TEXT_PROMPT="rotate_qa_after_sft"
PROMPT_FN="rotate_qa"
IMAGE_TOKEN_PROMPT="special_image_token"
DATASET="rotate-qa"
DATASET_PATH="~/datasets/ROTATE"
MODEL="InternVL2-8B-rotate"
MODEL_PATH="~/model_params/rotate/stage2/internvl2-8b/5000"
CONFIG_PATH="~/model_params/internvl2-8b"
PROCESSOR_PATH="~/model_params/internvl2-8b"
VIT_LORA_NUM=1
MLP_LORA_NUM=2
LLM_LORA_NUM=2
OUTPUT="test/${NAME}.json"

srun --unbuffered python evaluate.py -tp ${TEXT_PROMPT} \
                                     -d ${DATASET} \
                                     -dp ${DATASET_PATH} \
                                     -pf ${PROMPT_FN} \
                                     -itp ${IMAGE_TOKEN_PROMPT} \
                                     -m ${MODEL} \
                                     -mp ${MODEL_PATH} \
                                     -cp ${CONFIG_PATH} \
                                     -pp ${PROCESSOR_PATH} \
                                     --frozen_vit \
                                     --frozen_mlp \
                                     --frozen_llm \
                                     --frozen_lm_head \
                                     --vit_lora_num ${VIT_LORA_NUM} \
                                     --mlp_lora_num ${MLP_LORA_NUM} \
                                     --llm_lora_num ${LLM_LORA_NUM} \
                                     -o ${OUTPUT} \
                                     --use_flash_attention_2 \
                                     --fp16 