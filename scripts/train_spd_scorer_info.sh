#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOGURU_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
set -e  # 遇到错误立即退出

# ==============================================================================
# 环境清理与检查
# ==============================================================================
# 强制停止残留的 Ray 进程，防止连接到旧的集群导致 GPU 调度错误
echo "清理残留 Ray 进程..."
ray stop --force || true
pkill -f "ray" || true

# 检查 GPU 可见性
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print(f'Torch sees {torch.cuda.device_count()} GPUs')"

# ==============================================================================
# 配置参数
# ==============================================================================
export HF_HOME=/wekafs/jinzeli2/cache
export HF_HUB_OFFLINE=1
export WANDB_API_KEY="88b970302b89c7b55c90532cfd69ce4ee64ba81a"   # WandB API key，用于日志上报

# 模型配置
MODEL_PATH="meta-llama/Llama-3.1-8B"                      # 训练使用的基模型
TOKENIZER_PATH="meta-llama/Llama-3.1-8B-Instruct"         # Tokenizer 路径（基础模型需使用 Instruct 版本以支持 chat_template）
ADAPTER_PATH="/wekafs/jinzeli2/LLaMA-Factory/saves/llama3.1-8b/sft/251218exp0_llama3_8b_NoInstruct_NewTrainData_bs128_rank32_lr2e-4_4devices_warmup0.03_cutoff1024/checkpoint-20000" # LoRA Checkpoint 路径
LORA_RANK=32                                                      # LoRA 低秩分解的秩
LORA_ALPHA=64                                                     # LoRA scaling 系数
LORA_DROPOUT=0.0                                                  # LoRA Dropout
TARGET_MODULES="all"                                              # LoRA 目标模块

# 数据配置
DATA_DIR="/wekafs/jinzeli2/spec_boost/data"                        # 数据目录
SPD_GEN_DATA_FILE="/wekafs/jinzeli2/spec_boost/data/251130_trainRL_0_8000.jsonl"   # RL 训练数据（生成结果）
METADATA_FILE="/wekafs/jinzeli2/spec_boost/data/correct_samples_default_withids.jsonl" # 参考标签/元数据

# 训练配置
N_GPUS=4                                                        # 使用的 GPU 数量
TRAIN_BATCH_SIZE=64                                             # 全局 batch size
ROLLOUT_N=4                                                    # 每个 step 生成的 rollout 数
TOTAL_EPOCHS=2                                                  # 训练轮数
PPO_MINI_BATCH_SIZE=64                                           # PPO 小批次大小
micro_batch_size_per_gpu=16
MAX_PROMPT_LENGTH=1024                                         # 最长 prompt 长度 (只保留最后1024个token)
SEP_TOKEN_ID="eot"                                              # 分隔/结束 token id（字符串形式）

# 奖励配置
REWARD_ALPHA=1.0                                                # 长度奖励系数 (每接受一个 token 的奖励)
REWARD_PENALTY_BREAK=-5.0                                       # 场景B惩罚: 破坏原本正确的答案
REWARD_CORRECT_BASE=5.0                                         # 场景D基础奖励: 纠正错误的基础分
REWARD_USELESS=0.0                                              # 场景C奖励: 无用尝试 (原本错，Hybrid也错)

export SPD_VLLM_URLS="http://localhost:8000/v1/completions,http://localhost:8001/v1/completions,http://localhost:8002/v1/completions,http://localhost:8003/v1/completions"


# 日志配置
PROJECT_NAME="verl_spd_scorer"                                  # WandB 项目名
EXPERIMENT_NAME="251224exp1"                             # 实验名称

# ==============================================================================
# 启动训练
# ==============================================================================

echo "============================================================"
echo "SPD Scorer GRPO 训练"
echo "============================================================"

python train_spd_scorer.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --adapter_path ${ADAPTER_PATH} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --target_modules ${TARGET_MODULES} \
    --data_dir ${DATA_DIR} \
    --spd_gen_data_file ${SPD_GEN_DATA_FILE} \
    --metadata_file ${METADATA_FILE} \
    --n_gpus ${N_GPUS} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --micro_batch_size_per_gpu ${micro_batch_size_per_gpu} \
    --rollout_n ${ROLLOUT_N} \
    --total_epochs ${TOTAL_EPOCHS} \
    --ppo_mini_batch_size ${PPO_MINI_BATCH_SIZE} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --sep_token_id ${SEP_TOKEN_ID} \
    --reward_alpha ${REWARD_ALPHA} \
    --reward_penalty_break ${REWARD_PENALTY_BREAK} \
    --reward_correct_base ${REWARD_CORRECT_BASE} \
    --reward_useless ${REWARD_USELESS} \
    --project_name ${PROJECT_NAME} \
    --experiment_name ${EXPERIMENT_NAME} \
    &> /wekafs/jinzeli2/spec_boost/log/${EXPERIMENT_NAME}   
    # --offload                   # 可选: 启用 FSDP CPU Offload
    # --no_wandb \                 # 可选: 禁用 WandB
    # --overwrite_data \
