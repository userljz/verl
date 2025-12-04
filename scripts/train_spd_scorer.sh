#!/bin/bash
# ==============================================================================
# SPD Scorer GRPO 训练启动脚本
# 
# 使用方法:
#   bash scripts/train_spd_scorer.sh
#
# 注意: 运行前请根据实际情况修改下面的配置参数
# ==============================================================================

set -e  # 遇到错误立即退出

# ==============================================================================
# 配置参数
# ==============================================================================
export HF_HOME=/wekafs/jinzeli2/cache
export WANDB_API_KEY="88b970302b89c7b55c90532cfd69ce4ee64ba81a"

# 模型配置
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
TARGET_MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
LORA_RANK=16
LORA_ALPHA=32

# 数据配置
DATA_DIR="/wekafs/jinzeli2/spec_boost/data"
SPD_GEN_DATA_FILE="/wekafs/jinzeli2/spec_boost/data/251130_trainRL_0_8000.jsonl"
METADATA_FILE="/wekafs/jinzeli2/spec_boost/data/correct_samples_default_withids.jsonl"

# 训练配置
N_GPUS=8
TRAIN_BATCH_SIZE=64
ROLLOUT_N=8
TOTAL_EPOCHS=3
PPO_MINI_BATCH_SIZE=32
MICRO_BATCH_SIZE_PER_GPU=4
VLLM_GPU_MEMORY_UTILIZATION=0.7
SEP_TOKEN_ID="eot"

# 奖励配置
REWARD_ALPHA=1.0
REWARD_PENALTY_BREAK=-10.0
REWARD_CORRECT=100.0
REWARD_USELESS=0.0

# 日志配置
PROJECT_NAME="verl_spd_scorer"
EXPERIMENT_NAME="spd_grpo_training"

# ==============================================================================
# 启动训练
# ==============================================================================

echo "============================================================"
echo "SPD Scorer GRPO 训练"
echo "============================================================"

python train_spd_scorer.py \
    --model_path ${MODEL_PATH} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --data_dir ${DATA_DIR} \
    --spd_gen_data_file ${SPD_GEN_DATA_FILE} \
    --metadata_file ${METADATA_FILE} \
    --n_gpus ${N_GPUS} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --rollout_n ${ROLLOUT_N} \
    --total_epochs ${TOTAL_EPOCHS} \
    --ppo_mini_batch_size ${PPO_MINI_BATCH_SIZE} \
    --micro_batch_size_per_gpu ${MICRO_BATCH_SIZE_PER_GPU} \
    --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
    --sep_token_id ${SEP_TOKEN_ID} \
    --reward_alpha ${REWARD_ALPHA} \
    --reward_penalty_break ${REWARD_PENALTY_BREAK} \
    --reward_correct ${REWARD_CORRECT} \
    --reward_useless ${REWARD_USELESS} \
    --project_name ${PROJECT_NAME} \
    --experiment_name ${EXPERIMENT_NAME} \
    --target_model_path ${TARGET_MODEL_PATH} \
    --overwrite_data \
    &> /wekafs/jinzeli2/spec_boost/log/251204_test1   
    # --overwrite_data            # 可选: 强制覆盖已存在的训练数据
    # --offload                   # 可选: 启用 FSDP CPU Offload
    # --no_wandb                  # 可选: 禁用 WandB
