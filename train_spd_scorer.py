# -*- coding: utf-8 -*-
"""
Speculative Decoding Scorer 训练脚本
使用 verl 框架的 GRPO 算法训练 SPD Scoring Model

核心思想:
    - 将 SPD Scorer 的 Accept/Reject 决策建模为一种特殊的"序列生成"任务
    - 输入: [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]
    - 输出: 对每个 Draft Token 位置的 Accept/Reject 决策
    - Reward: 基于四场景逻辑计算 (加速成功/破坏正确/无用尝试/纠正错误)

作者: AI Assistant
日期: 2025-11-25
"""

import os
import sys
import subprocess
from typing import Optional, Dict, List, Tuple

import torch
import pandas as pd
from loguru import logger

from transformers import AutoTokenizer

# 确保当前目录在 sys.path 中
sys.path.insert(0, os.getcwd())

# ==============================================================================
# Monkey Patch: 注册 SPD Scorer 模型
# ==============================================================================
import verl.utils.model
from spd_scorer import AutoModelForSPDScoring

# 强制注入自定义的模型加载逻辑
_original_create_huggingface_actor = verl.utils.model.create_huggingface_actor

def _patched_create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> torch.nn.Module:
    """
    Hook 后的 create_huggingface_actor 函数，拦截并加载 SPD Scorer
    """
    if override_config_kwargs is None:
        override_config_kwargs = {}
    if automodel_kwargs is None:
        automodel_kwargs = {}
        
    logger.info(f"[Patch] Intercepting model loading for: {model_name}")
    logger.info(f"[Patch] Loading AutoModelForSPDScoring...")
        
    # 获取 HF Config
    module_config = verl.utils.model.get_huggingface_actor_config(
        model_name, override_config_kwargs, trust_remote_code=automodel_kwargs.get("trust_remote_code", False)
    )
    
    # 使用 SPD Scorer Factory 加载模型
    # 注意: 这里会调用 ScoringActor 的初始化，内部可能会再次加载 Backbone
    # 但由于 vLLM 和 HF 的缓存机制，或者单纯的多次加载，只要显存足够，是可以接受的
    model = AutoModelForSPDScoring.from_config(module_config, **automodel_kwargs)
    
    return model

# 应用 Patch: 替换 verl.utils.model 中的函数
verl.utils.model.create_huggingface_actor = _patched_create_huggingface_actor
logger.info("✅ 已应用 Monkey Patch: verl.utils.model.create_huggingface_actor -> AutoModelForSPDScoring")

# ==============================================================================
# 2. 数据准备
# ==============================================================================
def prepare_spd_data_from_real_source(
    args,
    source_data_path: List[str],
) -> Tuple[str, str]:
    """
    从真实数据源 (SPD生成数据 + Metadata) 准备 SPD 训练数据
    
    Args:
        args: 训练配置参数 (argparse.Namespace)
        source_data_path: 包含两个文件路径的列表 [spd_gen_data_file, metadata_file]
    """
    # 1. 计算目标路径
    train_file = args.train_file
    val_file = args.val_file
    
    train_path = os.path.join(args.data_dir, train_file)
    val_path = os.path.join(args.data_dir, val_file)
    
    # 2. 检查是否跳过 (Cache Hit)
    overwrite = getattr(args, "overwrite_data", False)
    if os.path.exists(train_path) and os.path.exists(val_path) and not overwrite:
        logger.info(f"训练数据已存在，跳过预处理步骤。")
        logger.info(f"Train: {train_path}")
        logger.info(f"Val:   {val_path}")
        return train_path, val_path

    if len(source_data_path) != 2:
        raise ValueError("source_data_path 必须是包含两个元素的列表: [spd_gen_data_file, metadata_file]")
    
    spd_gen_data_file = source_data_path[0]
    meta_file = source_data_path[1]
    logger.info(f"SPD Gen Data File: {spd_gen_data_file}")
    logger.info(f"Metadata File: {meta_file}")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info(f"Tokenizer 加载成功: {args.model_path}")
    
    # 1. 加载 Metadata 到内存字典 (Index -> Data)
    logger.info("加载 Metadata...")
    meta_df = pd.read_json(meta_file, lines=True)
    # index 字段即为 sample_idx
    meta_dict = meta_df.set_index("index").to_dict(orient="index")
    logger.info(f"Metadata 数据量: {len(meta_dict)}")
    
    # 2. 遍历 SPD 生成数据并处理
    logger.info("处理 SPD 生成数据...")
    spd_gen_df = pd.read_json(spd_gen_data_file, lines=True)
    logger.info(f"SPD 生成数据量: {len(spd_gen_df)}")
    
    os.makedirs(args.data_dir, exist_ok=True)
    processed_data = []
    
    for idx, row in spd_gen_df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"处理进度: {idx}/{len(spd_gen_df)}")
            
        sample_idx = row.get('sample_idx')
        cut_idx = row.get('cut_idx')
        
        # 查找 Metadata
        if sample_idx not in meta_dict:
            logger.warning(f"Sample {sample_idx} not found in metadata. Skipping.")
            continue
            
        meta = meta_dict[sample_idx]
        
        problem = meta.get('problem')
        ground_truth = meta.get('reference_answer')
        is_correct_baseline = meta.get('is_correct')
        
        # 构造 Context IDs
        # Step A: Base Chat (System + User)
        base_messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {
                "role": "user",
                "content": problem,
            },
        ]
        
        # apply_chat_template 返回 tensor if return_tensors='pt'
        # 这里我们需要 list[int] 以便拼接
        base_ids = tokenizer.apply_chat_template(
            base_messages,
            tokenize=True,
            add_generation_prompt=True
        )
        
        # Step B: Answer Prefix
        full_answer_ids = meta.get('answer_ids')
        answer_prefix_ids = full_answer_ids[:cut_idx]
        
        # Step C: Splice Context
        context_ids = base_ids + answer_prefix_ids
        
        # 获取 Draft / Target IDs
        draft_ids = row.get('draft_output_ids')
        target_ids = row.get('target_output_ids')
        
        # 严格校验长度: Target 应该比 Draft 多一个 bonus token
        if len(target_ids) != len(draft_ids) + 1:
            logger.warning(f"Sample {sample_idx} draft/target length mismatch: target ({len(target_ids)}) should be draft ({len(draft_ids)}) + 1. Skipping.")
            continue
        if len(draft_ids) == 0:
            logger.warning(f"Sample {sample_idx} draft length is 0. Skipping.")
            continue
            
        draft_len = len(draft_ids)
        target_len = len(target_ids) - 1  # 去掉 bonus token 后的长度，应该等于 draft_len

        # =================================================================
        # 构造完整的 Input IDs (用于 Actor 输入)
        # 结构: [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]
        # =================================================================
        
        # 获取 SEP Token ID (Llama-3 eot_id)
        if args.sep_token_id == "eot":
            sep_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.eot_token_id
        else:
            raise ValueError(f"Invalid sep_token_id: {args.sep_token_id}")
        
        # 拼接
        # 注意: 这里假设 draft_ids 和 target_ids 已经是 list[int]
        full_input_ids = (
            context_ids + 
            [sep_token_id] + 
            draft_ids + 
            [sep_token_id] + 
            target_ids[:-1] + 
            [sep_token_id]
        )
        
        # 计算关键位置索引 (用于后续 Mask 生成和逻辑处理)
        # draft_start_idx: Draft Tokens 的起始位置 (包含前面的 SEP)
        # 实际上在 list 索引中，draft_start_idx 指向的是 Draft 的第一个 Token
        # context_len (包含 SEP) = len(context_ids) + 1
        draft_start_idx = len(context_ids) + 1 
        
        # draft_end_idx: Draft Tokens 的结束位置 (不包含后面的 SEP)
        draft_end_idx = draft_start_idx + draft_len
        
        # target_start_idx: Target Tokens 的起始位置
        # 前面有: Context + SEP + Draft + SEP
        target_start_idx = draft_end_idx + 1
        
        # target_end_idx: Target Tokens 的结束位置
        target_end_idx = target_start_idx + target_len
            
        # 构造样本
        sample = {
            "data_source": "spd_scorer",
            
            # (1) 为了兼容 verl Dataset 接口，这里放一个 dummy prompt (实际上我们的 rollout/model 应该直接读取 input_ids)
            "prompt": "dummy_prompt", 
            
            # =====================================================
            # 核心数据 (顶层字段，方便后续 Rollout/Training 直接读取)
            # =====================================================
            "input_ids": full_input_ids,    # [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]
            
            # =====================================================
            # verl 兼容字段
            # =====================================================
            "ability": "spd_scoring",
            "reward_model": {
                "style": "rule",
                # (2) 将 Ground Truth 放入 reward_model 字段，这是 verl 的惯例
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "context_ids": context_ids,
                "draft_tokens": draft_ids,
                "target_tokens": target_ids[:-1],
                "bonus_tokens": target_ids[-1:],
                "is_correct_baseline": is_correct_baseline,
                "draft_len": draft_len,
                # 位置信息 (Non-Tensor, 但可以在 Rollout 中转为 Tensor)
                "draft_start_idx": draft_start_idx,
                "draft_end_idx": draft_end_idx,
                "target_start_idx": target_start_idx,
                "target_end_idx": target_end_idx,
            }
        }
        processed_data.append(sample)
    
    # 保存
    df = pd.DataFrame(processed_data)
    train_size = int(len(df) * 0.95)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    logger.info(f"处理完成: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")
    
    return train_path, val_path


# ==============================================================================
# 3. 训练主函数
# ==============================================================================

def build_training_command(args, train_file: str, val_file: str) -> list:
    """
    构建 verl GRPO 训练命令
    
    Args:
        args: 训练配置参数 (argparse.Namespace)
        train_file: 训练数据路径
        val_file: 验证数据路径
    
    Returns:
        cmd: 训练命令列表
    """
    
    # 基础命令
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # =================================================================
        # 算法核心配置 (GRPO)
        # =================================================================
        "algorithm.adv_estimator=grpo",        # 使用 GRPO 算法
        "algorithm.use_kl_in_reward=False",    # GRPO 特性
        "algorithm.kl_ctrl.kl_coef=0.001",     # KL 散度系数
        
        # =================================================================
        # 数据配置
        # =================================================================
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        
        # 使用自定义的 SPD Dataset（通过 pkg:// 按模块导入）
        "data.custom_cls.path=pkg://verl.utils.dataset.spd_dataset",
        "data.custom_cls.name=SPDRLHFDataset",
        
        f"data.train_batch_size={args.train_batch_size}",
        "data.max_prompt_length=2048",         # 包含 Context + Draft + Target
        "data.max_response_length=50",        # 输出是 Accept/Reject 决策序列
        
        # =================================================================
        # 模型配置
        # =================================================================
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        
        # LoRA 配置
        f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        
        # =================================================================
        # Rollout 配置
        # =================================================================
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        "actor_rollout_ref.rollout.name=spd",  # 使用自定义的 SPD Rollout
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.vllm_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        f"actor_rollout_ref.rollout.data_parallel_size={args.n_gpus}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        
        # =================================================================
        # Actor 训练配置
        # =================================================================
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={args.offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={args.offload}",
        
        # =================================================================
        # Reference 配置
        # =================================================================
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={args.offload}",
        
        # =================================================================
        # Trainer 配置
        # =================================================================
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        "trainer.test_freq=10",
        "trainer.save_freq=-1",
    ]
    
    # 日志配置
    if not args.no_wandb:
        cmd.append("trainer.logger=['console','wandb']")
    else:
        cmd.append("trainer.logger=['console']")
    
    return cmd


def run_training(args, source_data_path: List[str]):
    """
    运行 SPD Scorer 训练
    
    完整流程:
        1. 准备训练数据
        2. 构建并执行训练命令
    
    Args:
        args: 训练配置参数 (argparse.Namespace)
        source_data_path: 包含两个文件路径的列表 [spd_gen_data_file, metadata_file]
    """
    logger.info("=" * 60)
    logger.info("SPD Scorer GRPO 训练")
    logger.info("=" * 60)
    
    # Step 1: 准备数据
    logger.info("\n[Step 1] 准备训练数据...")
    train_file, val_file = prepare_spd_data_from_real_source(args, source_data_path)
    
    # Step 2: 构建训练命令
    logger.info("\n[Step 2] 构建训练命令...")
    cmd = build_training_command(args, train_file, val_file)
    
    logger.info("\n训练命令:")
    logger.info(" ".join(cmd[:5]) + " \\")
    for arg in cmd[5:]:
        logger.info(f"    {arg} \\")
    
    # Step 3: 执行训练
    logger.info("\n[Step 3] 启动训练...")
    logger.info("=" * 60)
    
    logger.info(f"🚀 开始 SPD Scorer GRPO 训练")
    logger.info(f"配置: {args.n_gpus} GPU | Batch={args.train_batch_size} | Rollout N={args.rollout_n} | Offload={'ON' if args.offload else 'OFF'}")
    logger.info("=" * 60)
    
    # 创建训练环境变量
    env = _create_training_env(args)
    
    subprocess.run(cmd, check=True, env=env)
        
 

def _create_training_env(args) -> Dict[str, str]:
    """
    生成训练子进程所需的环境变量
    
    包含:
    1. Reward Function 配置
    2. Model 路径配置 (用于 spd_scorer 和 spd_scorer_reward)
    3. 分布式训练配置
    """
    env = os.environ.copy()
    
    # 1. Reward 配置
    env["SPD_REWARD_ALPHA"] = str(args.reward_alpha)
    env["SPD_REWARD_PENALTY_BREAK"] = str(args.reward_penalty_break)
    env["SPD_REWARD_CORRECT"] = str(args.reward_correct)
    env["SPD_REWARD_USELESS"] = str(args.reward_useless)
    
    # 2. 模型配置 (注入环境变量，供 spd_scorer.py 和 spd_scorer_reward.py 读取)
    env["SPD_MODEL_PATH"] = str(args.model_path)
    
    # Target Model Path: 优先使用 target_model_path，如果没有指定，fallback 到 model_path
    env["SPD_TARGET_MODEL_PATH"] = str(args.target_model_path)
   
        
    env["SPD_LORA_RANK"] = str(args.lora_rank)
    env["SPD_LORA_ALPHA"] = str(args.lora_alpha)
    
    # 处理 SEP Token ID
    # 如果是 "eot"，则尝试加载 tokenizer 解析，或者使用默认值
    if str(args.sep_token_id).lower() == "eot":
        logger.info(f"解析 sep_token_id='eot'...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # 优先尝试 eot_id (Llama-3), 然后 eos_token_id
        if hasattr(tokenizer, "eot_token_id") and tokenizer.eot_token_id is not None:
            real_sep_id = tokenizer.eot_token_id
        
        else:
            raise ValueError(f"无法解析 sep_token_id: {args.sep_token_id}")
        logger.info(f"已解析 sep_token_id: eot -> {real_sep_id}")
        env["SPD_SEP_TOKEN_ID"] = str(real_sep_id)
        
    else:
        raise ValueError(f"Invalid sep_token_id: {args.sep_token_id}")
    
    # 3. 基础训练配置
    env["HYDRA_FULL_ERROR"] = "1"
    env["NCCL_P2P_DISABLE"] = "1"
    
    return env


# ==============================================================================
# 5. 命令行入口
# ==============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPD Scorer GRPO 训练")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3-8B", help="基础模型路径")
    parser.add_argument("--target_model_path", type=str, default=None, help="Target 模型路径 (用于 Reward Tokenizer)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default="data/spd_scorer", help="数据目录")
    parser.add_argument("--train_file", type=str, default="train.parquet", help="训练数据文件名")
    parser.add_argument("--val_file", type=str, default="val.parquet", help="验证数据文件名")
    parser.add_argument("--spd_gen_data_file", type=str, required=True, help="SPD 生成数据文件路径 (jsonl)")
    parser.add_argument("--metadata_file", type=str, required=True, help="Metadata 文件路径 (jsonl)")
    parser.add_argument("--overwrite_data", action="store_true", help="是否强制覆盖已存在的训练数据")
    
    # 训练配置
    parser.add_argument("--n_gpus", type=int, default=8, help="GPU 数量")
    parser.add_argument("--train_batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--rollout_n", type=int, default=8, help="GRPO Rollout N")
    parser.add_argument("--total_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=32, help="PPO mini batch 大小")
    parser.add_argument("--micro_batch_size_per_gpu", type=int, default=4, help="每 GPU 微批次大小")
    parser.add_argument("--offload", action="store_true", help="启用 FSDP CPU Offload (省显存但降低速度)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7, help="vLLM GPU 显存利用率")
    parser.add_argument("--sep_token_id", type=str, default="eot", help="分隔符 Token ID (eot 或数字)")
    
    # 奖励配置
    parser.add_argument("--reward_alpha", type=float, default=1.0, help="场景A奖励系数")
    parser.add_argument("--reward_penalty_break", type=float, default=-10.0, help="场景B惩罚")
    parser.add_argument("--reward_correct", type=float, default=100.0, help="场景D奖励")
    parser.add_argument("--reward_useless", type=float, default=0.0, help="场景C奖励")
    
    # 其他
    parser.add_argument("--no_wandb", action="store_true", help="禁用 WandB")
    parser.add_argument("--project_name", type=str, default="verl_spd_scorer", help="WandB 项目名")
    parser.add_argument("--experiment_name", type=str, default="spd_grpo_training", help="实验名")
    
    args = parser.parse_args()
    
    # 打印配置
    logger.info("\n" + "=" * 60)
    logger.info("SPD Scorer 训练配置")
    logger.info("=" * 60)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60 + "\n")
    
    source_data_path = [args.spd_gen_data_file, args.metadata_file]
    run_training(args, source_data_path)
    


if __name__ == "__main__":
    main()

