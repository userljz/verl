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
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import pandas as pd

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 配置定义
# ==============================================================================

@dataclass
class SPDTrainingConfig:
    """SPD Scorer 训练配置"""
    
    # 模型路径
    model_path: str = "meta-llama/Llama-3-8B"
    
    # 数据路径
    data_dir: str = "data/spd_scorer"
    train_file: str = "train.parquet"
    val_file: str = "val.parquet"
    
    # LoRA 配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # 训练超参数
    n_gpus: int = 8
    train_batch_size: int = 64
    ppo_mini_batch_size: int = 32
    micro_batch_size_per_gpu: int = 4
    rollout_n: int = 8  # GRPO: 每个样本采样 N 个决策序列
    total_epochs: int = 3
    
    # 奖励系数
    reward_alpha: float = 1.0          # 场景 A: alpha * L
    reward_penalty_break: float = -10.0  # 场景 B: 破坏正确答案
    reward_correct: float = 100.0       # 场景 D: 纠正错误
    reward_useless: float = 0.0         # 场景 C: 无用尝试
    
    # 特殊 Token
    sep_token: str = "<|sep|>"
    sep_token_id: int = 128009  # Llama-3 的 <|eot_id|>，可根据实际情况调整
    
    # 补全服务配置
    target_model_url: Optional[str] = None  # e.g. "http://localhost:8000/v1/completions"
    target_model_name: str = "target-model"
    
    # 其他配置
    vllm_gpu_memory_utilization: float = 0.7
    offload: bool = False
    use_wandb: bool = True
    project_name: str = "verl_spd_scorer"
    experiment_name: str = "spd_grpo_training"


# ==============================================================================
# 2. 数据准备
# ==============================================================================
def prepare_spd_data_from_real_source(
    config: SPDTrainingConfig,
    source_data_path: List[str],
) -> Tuple[str, str]:
    """
    从真实数据源 (SPD生成数据 + Metadata) 准备 SPD 训练数据
    
    Args:
        config: 训练配置
        source_data_path: 包含两个文件路径的列表 [spd_gen_data_file, metadata_file]
    """
    if len(source_data_path) != 2:
        raise ValueError("source_data_path 必须是包含两个元素的列表: [spd_gen_data_file, metadata_file]")
    
    spd_gen_data_file = source_data_path[0]
    meta_file = source_data_path[1]
    logger.info(f"SPD Gen Data File: {spd_gen_data_file}")
    logger.info(f"Metadata File: {meta_file}")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    logger.info(f"Tokenizer 加载成功: {config.model_path}")
    
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
    
    os.makedirs(config.data_dir, exist_ok=True)
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
        
        # 严格校验长度: 如果 Draft 和 Target 长度不一致，直接跳过
        if len(draft_ids) != len(target_ids):
            logger.warning(f"Sample {sample_idx} draft/target length mismatch ({len(draft_ids)} vs {len(target_ids)}). Skipping.")
            continue
        if len(draft_ids) == 0:
            logger.warning(f"Sample {sample_idx} draft length is 0. Skipping.")
            continue
            
        draft_len = len(draft_ids)
        target_len = len(target_ids) # 应该等于 draft_len

        # =================================================================
        # 构造完整的 Input IDs (用于 Actor 输入)
        # 结构: [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]
        # =================================================================
        
        # 获取 SEP Token ID (Llama-3 eot_id)
        sep_token_id = config.sep_token_id
        
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
    
    train_path = os.path.join(config.data_dir, config.train_file)
    val_path = os.path.join(config.data_dir, config.val_file)
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    logger.info(f"处理完成: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")
    
    return train_path, val_path


# ==============================================================================
# 3. 训练主函数
# ==============================================================================

def build_training_command(config: SPDTrainingConfig, train_file: str, val_file: str) -> list:
    """
    构建 verl GRPO 训练命令
    
    Args:
        config: 训练配置
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
        
        # 使用自定义的 SPD Dataset
        "data.custom_cls.path=verl.utils.dataset.spd_dataset",
        "data.custom_cls.name=SPDRLHFDataset",
        
        f"data.train_batch_size={config.train_batch_size}",
        "data.max_prompt_length=2048",         # 包含 Context + Draft + Target
        "data.max_response_length=256",        # 输出是 Accept/Reject 决策序列
        
        # =================================================================
        # 模型配置
        # =================================================================
        f"actor_rollout_ref.model.path={config.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        
        # LoRA 配置
        f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
        
        # =================================================================
        # Rollout 配置
        # =================================================================
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        "actor_rollout_ref.rollout.name=spd",  # 使用自定义的 SPD Rollout
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.vllm_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        f"actor_rollout_ref.rollout.data_parallel_size={config.n_gpus}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config.micro_batch_size_per_gpu}",
        
        # =================================================================
        # Actor 训练配置
        # =================================================================
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={config.offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={config.offload}",
        
        # =================================================================
        # Reference 配置
        # =================================================================
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={config.offload}",
        
        # =================================================================
        # Trainer 配置
        # =================================================================
        f"trainer.total_epochs={config.total_epochs}",
        f"trainer.n_gpus_per_node={config.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.project_name={config.project_name}",
        f"trainer.experiment_name={config.experiment_name}",
        "trainer.test_freq=10",
        "trainer.save_freq=-1",
    ]
    
    # 日志配置
    if config.use_wandb:
        cmd.append("trainer.logger=['console','wandb']")
    else:
        cmd.append("trainer.logger=['console']")
    
    return cmd


def run_training(config: SPDTrainingConfig, source_data_path: List[str]):
    """
    运行 SPD Scorer 训练
    
    完整流程:
        1. 准备训练数据
        2. 构建并执行训练命令
    
    Args:
        config: 训练配置
        source_data_path: 包含两个文件路径的列表 [spd_gen_data_file, metadata_file]
    """
    logger.info("=" * 60)
    logger.info("SPD Scorer GRPO 训练")
    logger.info("=" * 60)
    
    # Step 1: 准备数据
    logger.info("\n[Step 1] 准备训练数据...")
    train_file, val_file = prepare_spd_data_from_real_source(config, source_data_path)
    
    # Step 2: 构建训练命令
    logger.info("\n[Step 2] 构建训练命令...")
    cmd = build_training_command(config, train_file, val_file)
    
    logger.info("\n训练命令:")
    logger.info(" ".join(cmd[:5]) + " \\")
    for arg in cmd[5:]:
        logger.info(f"    {arg} \\")
    
    # Step 3: 执行训练
    logger.info("\n[Step 3] 启动训练...")
    logger.info("=" * 60)
    
    offload_status = "ON" if config.offload else "OFF"
    logger.info(f"🚀 开始 SPD Scorer GRPO 训练")
    logger.info(f"配置: {config.n_gpus} GPU | Batch={config.train_batch_size} | Rollout N={config.rollout_n} | Offload={offload_status}")
    logger.info("=" * 60)
    
    try:
        # 获取包含奖励配置的环境变量
        env = _get_reward_config_env(config)
        
        # [NEW] 将模型配置也注入环境变量，供 spd_scorer.py 读取
        env["SPD_MODEL_PATH"] = str(config.model_path)
        env["SPD_LORA_RANK"] = str(config.lora_rank)
        env["SPD_LORA_ALPHA"] = str(config.lora_alpha)
        
        env["HYDRA_FULL_ERROR"] = "1"
        env["NCCL_P2P_DISABLE"] = "1"
        
        subprocess.run(cmd, check=True, env=env)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n训练过程中出错: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断。")


def _get_reward_config_env(config: SPDTrainingConfig) -> Dict[str, str]:
    """生成 Reward Function 所需的环境变量"""
    env = os.environ.copy()
    env["SPD_REWARD_ALPHA"] = str(config.reward_alpha)
    env["SPD_REWARD_PENALTY_BREAK"] = str(config.reward_penalty_break)
    env["SPD_REWARD_CORRECT"] = str(config.reward_correct)
    env["SPD_REWARD_USELESS"] = str(config.reward_useless)
    if config.target_model_url:
        env["SPD_TARGET_MODEL_URL"] = str(config.target_model_url)
    if config.target_model_name:
        env["SPD_TARGET_MODEL_NAME"] = str(config.target_model_name)
    return env


# ==============================================================================
# 4. 独立的 SPD Scorer 训练循环 (不依赖 verl 的 main_ppo)
# ==============================================================================

def train_spd_scorer_standalone(config: SPDTrainingConfig):
    """
    独立的 SPD Scorer 训练函数
    
    这个函数提供了一个不完全依赖 verl.trainer.main_ppo 的训练选项。
    它直接使用 verl 的底层组件 (FSDP Engine, Optimizer 等) 来训练 SPD Scorer。
    
    适用场景:
        - 需要更精细地控制训练流程
        - SPD Scorer 的输出格式与标准 LLM 差异较大
        - 需要自定义 rollout 逻辑
    
    Args:
        config: 训练配置
    """
    logger.info("=" * 60)
    logger.info("SPD Scorer 独立训练模式")
    logger.info("=" * 60)
    
    # 导入必要的模块
    try:
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader
        
        # 导入 SPD Scorer
        from spd_scorer import ScoringActor, ScoringModelConfig, SPDRewardFunction
        
        logger.info("成功导入 SPD Scorer 模块")
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.info("请确保 spd_scorer.py 在当前目录")
        return
    
    # 初始化分布式环境 (如果需要)
    if not dist.is_initialized():
        # 单机训练时，使用简单的初始化
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            logger.warning("CUDA 不可用，使用 CPU 训练")
    
    # 创建模型配置
    model_config = ScoringModelConfig(
        model_name_or_path=config.model_path,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
    )
    
    # 创建奖励函数
    reward_fn = SPDRewardFunction(
        alpha=config.reward_alpha,
        penalty_break=config.reward_penalty_break,
        reward_correct=config.reward_correct,
        reward_useless=config.reward_useless,
    )
    
    logger.info("\n模型配置:")
    logger.info(f"  - 基础模型: {config.model_path}")
    logger.info(f"  - LoRA Rank: {config.lora_rank}")
    logger.info(f"  - LoRA Alpha: {config.lora_alpha}")
    
    logger.info("\n奖励配置:")
    logger.info(f"  - Alpha (场景A): {config.reward_alpha}")
    logger.info(f"  - Penalty Break (场景B): {config.reward_penalty_break}")
    logger.info(f"  - Reward Correct (场景D): {config.reward_correct}")
    logger.info(f"  - Reward Useless (场景C): {config.reward_useless}")
    
    # 这里只是一个框架示例
    # 完整实现需要:
    # 1. 加载数据
    # 2. 初始化模型 (ScoringActor)
    # 3. 实现 GRPO 训练循环
    # 4. 保存模型
    
    logger.info("\n[注意] 独立训练模式需要更多实现工作")
    logger.info("建议先使用 verl 集成模式 (run_training)")
    logger.info("如果需要完整的独立训练循环，请参考 verl 的 trainer 实现")


# ==============================================================================
# 5. 命令行入口
# ==============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPD Scorer GRPO 训练")
    
    parser.add_argument("--mode", type=str, default="verl", 
                        choices=["verl", "standalone"],
                        help="训练模式: verl (使用 verl 框架) 或 standalone (独立训练)")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3-8B",
                        help="基础模型路径")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default="data/spd_scorer",
                        help="数据目录")
    parser.add_argument("--spd_gen_data_file", type=str, required=True,
                        help="SPD 生成数据文件路径 (jsonl)")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Metadata 文件路径 (jsonl)")
    
    # 训练配置
    parser.add_argument("--n_gpus", type=int, default=8, help="GPU 数量")
    parser.add_argument("--train_batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--rollout_n", type=int, default=8, help="GRPO Rollout N")
    parser.add_argument("--total_epochs", type=int, default=3, help="训练轮数")
    
    # 奖励配置
    parser.add_argument("--reward_alpha", type=float, default=1.0, help="场景A奖励系数")
    parser.add_argument("--reward_penalty_break", type=float, default=-10.0, help="场景B惩罚")
    parser.add_argument("--reward_correct", type=float, default=100.0, help="场景D奖励")
    parser.add_argument("--reward_useless", type=float, default=0.0, help="场景C奖励")
    
    # 补全服务
    parser.add_argument("--target_model_url", type=str, default=None,
                        help="Target Model vLLM API 地址 (e.g. http://localhost:8000/v1/completions)")
    parser.add_argument("--target_model_name", type=str, default="target-model",
                        help="Target Model 名称")
    
    # 其他
    parser.add_argument("--no_wandb", action="store_true", help="禁用 WandB")
    parser.add_argument("--project_name", type=str, default="verl_spd_scorer",
                        help="WandB 项目名")
    parser.add_argument("--experiment_name", type=str, default="spd_grpo_training",
                        help="实验名")
    
    args = parser.parse_args()
    
    # 创建配置
    config = SPDTrainingConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        n_gpus=args.n_gpus,
        train_batch_size=args.train_batch_size,
        rollout_n=args.rollout_n,
        total_epochs=args.total_epochs,
        reward_alpha=args.reward_alpha,
        reward_penalty_break=args.reward_penalty_break,
        reward_correct=args.reward_correct,
        reward_useless=args.reward_useless,
        target_model_url=args.target_model_url,
        target_model_name=args.target_model_name,
        use_wandb=not args.no_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
    )
    
    # 打印配置
    logger.info("\n" + "=" * 60)
    logger.info("SPD Scorer 训练配置")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"模型: {config.model_path}")
    logger.info(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    logger.info(f"训练: {config.n_gpus} GPU, batch={config.train_batch_size}, epochs={config.total_epochs}")
    logger.info(f"GRPO: rollout_n={config.rollout_n}")
    logger.info(f"奖励: A={config.reward_alpha}*L, B={config.reward_penalty_break}, C={config.reward_useless}, D={config.reward_correct}")
    logger.info("=" * 60 + "\n")
    
    # 根据模式选择训练方法
    source_data_path = [args.spd_gen_data_file, args.metadata_file]
    
    if args.mode == "verl":
        run_training(config, source_data_path)
    else:
        train_spd_scorer_standalone(config)


if __name__ == "__main__":
    main()

