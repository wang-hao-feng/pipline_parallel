#!/bin/env zsh
#SBATCH -J internvl2-8b-stage1
#SBATCH -o output/train/stage1/%x.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50G
#SBATCH -t 24:00:00
#SBATCH -p compute 
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4

NODE_NUM=4

source ~/.zshrc

cd ~/rotate

conda activate torch25

TEXT_PROMPT="internvl2_rotate_caption"
PROMPT_FN="rotate_caption"
IMAGE_TOKEN_PROMPT="special_image_token"

DATASET="rotate-caption"
DATASET_PATH="~/datasets/ROTATE"
COLLATE_FN="sft"

MODEL="InternVL2-8B"
MODEL_PATH="~/model_params/internvl2-8b"
CONFIG_PATH="~/model_params/internvl2-8b"
PROCESSOR_PATH="~/model_params/internvl2-8b"
RESULT_PATH="~/model_params/rotate/stage1/internvl2-8b"
VIT_LORA_NUM=1
MLP_LORA_NUM=1
LLM_LORA_NUM=1

BATCH_SIZE=64
MICRO_BATCH_SIZE=4
MAX_TOKEN_NUM=1024
LEARNING_RATE=4e-5
WEIGHT_DECAY=0.05
MIN_LR=2e-5
WARM_UP_START_LR=3e-5
WARM_UP_STEP=100
EVAL_STEP=500
SAVE_STEP=500
TOTAL_STEP=5000
TASK="sft"
NEGATIVE_NUM=2

srun --unbuffered torchrun --nproc_per_node ${NODE_NUM} \
                           --master_port 6666 \
                           train.py -tp ${TEXT_PROMPT} \
                                    -pf ${PROMPT_FN} \
                                    -itp ${IMAGE_TOKEN_PROMPT} \
                                    -d ${DATASET} \
                                    -dp ${DATASET_PATH} \
                                    -cf ${COLLATE_FN} \
                                    -m ${MODEL} \
                                    -mp ${MODEL_PATH} \
                                    -cp ${CONFIG_PATH} \
                                    -pp ${PROCESSOR_PATH} \
                                    -r ${RESULT_PATH} \
                                    --vit_lora_num ${VIT_LORA_NUM} \
                                    --mlp_lora_num ${MLP_LORA_NUM} \
                                    --llm_lora_num ${LLM_LORA_NUM} \
                                    -bs ${BATCH_SIZE} \
                                    -mbs ${MICRO_BATCH_SIZE} \
                                    --max_token_num ${MAX_TOKEN_NUM} \
                                    -lr ${LEARNING_RATE}\
                                    -wd ${WEIGHT_DECAY}\
                                    -ml ${MIN_LR}\
                                    -wusl ${WARM_UP_START_LR}\
                                    -ws ${WARM_UP_STEP}\
                                    -es ${EVAL_STEP}\
                                    -ss ${SAVE_STEP}\
                                    -ts ${TOTAL_STEP}\
                                    -t ${TASK} \
                                    --sim_fn ${SIM} \
                                    -nn ${NEGATIVE_NUM} \
                                    --use_flash_attention_2