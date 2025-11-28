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
import json
import subprocess
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import random

import torch
import pandas as pd

# 确保当前目录在 sys.path 中
sys.path.insert(0, os.getcwd())

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

def prepare_spd_training_data(
    config: SPDTrainingConfig,
    num_samples: int = 10000,
    max_context_len: int = 512,
    max_draft_len: int = 32,
    seed: int = 42
) -> Tuple[str, str]:
    """
    准备 SPD Scorer 的训练数据
    
    数据格式说明:
        - 每个样本包含: Context, Draft Tokens, Target Tokens
        - 训练目标: 学习哪些 Mismatch 的 Draft Token 应该被接受
    
    实际使用时，你应该从真实的 Speculative Decoding 场景中收集数据:
        1. 使用 Draft Model 生成 draft tokens
        2. 使用 Target Model 验证并生成 target tokens
        3. 记录最终答案是否正确
    
    Args:
        config: 训练配置
        num_samples: 生成的样本数量
        max_context_len: 最大上下文长度
        max_draft_len: 最大 Draft 长度
        seed: 随机种子
    
    Returns:
        train_path, val_path: 训练和验证数据路径
    """
    logger.info(f"准备 SPD 训练数据，目标目录: {config.data_dir}")
    os.makedirs(config.data_dir, exist_ok=True)
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 生成模拟数据
    # 注意: 实际使用时，这里应该加载真实的 Speculative Decoding 数据
    processed_data = []
    
    for i in range(num_samples):
        if i % 1000 == 0:
            logger.info(f"已生成 {i}/{num_samples} 条数据...")
        
        # 模拟数据生成
        # 实际使用时，这些应该来自真实的 Draft/Target Model 输出
        draft_len = random.randint(8, max_draft_len)
        
        # 模拟 Context (这里用占位符，实际应该是真实文本)
        context_text = f"Context for sample {i}. " * random.randint(1, 5)
        
        # 模拟 Draft 和 Target tokens (用 token ID 表示)
        # 实际使用时，这些应该是真实的 token IDs
        draft_tokens = [random.randint(1000, 30000) for _ in range(draft_len)]
        
        # Target tokens: 与 Draft 有一定重合 (约 70% 相同)
        target_tokens = []
        for dt in draft_tokens:
            if random.random() < 0.7:
                target_tokens.append(dt)  # Match
            else:
                target_tokens.append(random.randint(1000, 30000))  # Mismatch
        
        # 模拟 Baseline 正确性 (Target Model 单独是否答对)
        is_correct_baseline = random.random() < 0.6  # 约 60% 正确率
        
        # 模拟 Ground Truth (最终正确答案)
        ground_truth = f"answer_{i % 100}"
        
        # 构造 verl 协议格式的数据
        # 关键: prompt 构造为 [Context + SEP + Draft + SEP + Target + SEP]
        sample = {
            # data_source 用于指定 reward function
            # 我们使用自定义的 "spd_scorer" data_source
            "data_source": "spd_scorer",
            
            # Prompt: 使用 Chat 格式
            # 实际输入会在 tokenize 后变成 [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]
            "prompt": [
                {
                    "role": "system", 
                    "content": "You are a scoring model for speculative decoding. "
                               "Decide which draft tokens to accept."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "context": context_text,
                        "draft_tokens": draft_tokens,
                        "target_tokens": target_tokens,
                    })
                }
            ],
            
            "ability": "spd_scoring",
            
            # reward_model 字段: 包含计算 reward 所需的所有信息
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "is_correct_baseline": is_correct_baseline,
                "draft_len": draft_len,
                # 奖励参数
                "alpha": config.reward_alpha,
                "penalty_break": config.reward_penalty_break,
                "reward_correct": config.reward_correct,
                "reward_useless": config.reward_useless,
            },
            
            # 额外信息
            "extra_info": {
                "split": "train",
                "index": i,
                "draft_len": draft_len,
                "match_ratio": sum(1 for d, t in zip(draft_tokens, target_tokens) if d == t) / draft_len,
                # 传递给 Reward Function 的关键信息
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "is_correct_baseline": is_correct_baseline,
                "alpha": config.reward_alpha,
                "penalty_break": config.reward_penalty_break,
                "reward_correct": config.reward_correct,
                "reward_useless": config.reward_useless,
                # 上下文和补全服务配置
                "context_text": context_text,
                "target_model_url": config.target_model_url,
                "target_model_name": config.target_model_name,
                "model_path": config.model_path,
            }
        }
        
        processed_data.append(sample)
    
    # 转换为 DataFrame
    df = pd.DataFrame(processed_data)
    
    # 划分训练集和验证集 (95% 训练, 5% 验证)
    train_size = int(len(df) * 0.95)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # 保存为 Parquet 文件
    train_path = os.path.join(config.data_dir, config.train_file)
    val_path = os.path.join(config.data_dir, config.val_file)
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    logger.info(f"数据已保存: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")
    logger.info(f"  - 训练集路径: {train_path}")
    logger.info(f"  - 验证集路径: {val_path}")
    
    return train_path, val_path


def prepare_spd_data_from_real_source(
    config: SPDTrainingConfig,
    source_data_path: str,
    tokenizer_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    从真实数据源准备 SPD 训练数据
    
    期望的输入数据格式 (JSON/Parquet):
    {
        "context": "...",           # 上下文文本
        "draft_response": "...",    # Draft Model 的输出
        "target_response": "...",   # Target Model 的输出
        "ground_truth": "...",      # 正确答案
        "is_correct_baseline": bool # Target Model 是否答对
    }
    
    Args:
        config: 训练配置
        source_data_path: 源数据路径
        tokenizer_path: Tokenizer 路径 (用于 tokenize 文本)
    
    Returns:
        train_path, val_path: 处理后的数据路径
    """
    logger.info(f"从真实数据源加载: {source_data_path}")
    
    # 加载 tokenizer
    if tokenizer_path is None:
        tokenizer_path = config.model_path
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    logger.info(f"Tokenizer 加载成功: {tokenizer_path}")
    
    # 加载源数据
    if source_data_path.endswith('.parquet'):
        source_df = pd.read_parquet(source_data_path)
    elif source_data_path.endswith('.json') or source_data_path.endswith('.jsonl'):
        source_df = pd.read_json(source_data_path, lines=source_data_path.endswith('.jsonl'))
    else:
        raise ValueError(f"不支持的数据格式: {source_data_path}")
    
    logger.info(f"加载了 {len(source_df)} 条源数据")
    
    os.makedirs(config.data_dir, exist_ok=True)
    processed_data = []
    
    for idx, row in source_df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"处理进度: {idx}/{len(source_df)}")
        
        # 提取字段
        context = row.get('context', '')
        draft_response = row.get('draft_response', '')
        target_response = row.get('target_response', '')
        ground_truth = row.get('ground_truth', '')
        is_correct_baseline = row.get('is_correct_baseline', False)
        
        # Tokenize
        draft_tokens = tokenizer.encode(draft_response, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_response, add_special_tokens=False)
        
        # 对齐长度 (取较短的)
        min_len = min(len(draft_tokens), len(target_tokens))
        draft_tokens = draft_tokens[:min_len]
        target_tokens = target_tokens[:min_len]
        
        if min_len == 0:
            continue
        
        # 构造样本
        sample = {
            "data_source": "spd_scorer",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a scoring model for speculative decoding."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "context": context,
                        "draft_tokens": draft_tokens,
                        "target_tokens": target_tokens,
                    })
                }
            ],
            "ability": "spd_scoring",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "is_correct_baseline": is_correct_baseline,
                "draft_len": min_len,
                "alpha": config.reward_alpha,
                "penalty_break": config.reward_penalty_break,
                "reward_correct": config.reward_correct,
                "reward_useless": config.reward_useless,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "draft_len": min_len,
                # 传递给 Reward Function 的关键信息
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "is_correct_baseline": is_correct_baseline,
                "alpha": config.reward_alpha,
                "penalty_break": config.reward_penalty_break,
                "reward_correct": config.reward_correct,
                "reward_useless": config.reward_useless,
                # 上下文和补全服务配置
                "context_text": context,
                "target_model_url": config.target_model_url,
                "target_model_name": config.target_model_name,
                "model_path": config.model_path,
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
# 3. 自定义 Reward Function
# ==============================================================================

def register_spd_reward_function():
    """
    注册 SPD Scorer 的自定义 Reward Function 到 verl
    
    这个函数会在 verl 启动前被调用，确保 reward function 可用
    """
    
    # 创建 reward function 文件
    reward_fn_code = '''
# -*- coding: utf-8 -*-
"""
SPD Scorer 自定义 Reward Function
用于 verl 框架的 reward 计算

奖励逻辑:
    - 场景 A (加速成功): Baseline 对 & Hybrid 对 -> Reward = alpha * L
    - 场景 B (破坏正确): Baseline 对 & Hybrid 错 -> Reward = penalty_break
    - 场景 C (无用尝试): Baseline 错 & Hybrid 错 -> Reward = reward_useless
    - 场景 D (纠正错误): Baseline 错 & Hybrid 对 -> Reward = reward_correct
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_effective_length(accept_decisions: list) -> int:
    """
    计算有效接受长度 L
    
    定义: 从序列开头连续为 True/1 的长度
    
    Args:
        accept_decisions: Accept/Reject 决策列表
    
    Returns:
        L: 有效接受长度
    """
    L = 0
    for decision in accept_decisions:
        if decision:
            L += 1
        else:
            break
    return L


def verify_hybrid_correctness(
    draft_tokens: list,
    target_tokens: list,
    accept_decisions: list,
    ground_truth: str
) -> bool:
    """
    验证 Hybrid 生成结果的正确性
    
    Hybrid 生成逻辑:
        - 接受的位置: 使用 Draft Token
        - 拒绝的位置: 使用 Target Token
    
    简化实现: 
        - 这里使用启发式规则判断正确性
        - 实际使用时应该调用真正的验证函数
    
    Args:
        draft_tokens: Draft token IDs
        target_tokens: Target token IDs
        accept_decisions: Accept/Reject 决策
        ground_truth: 正确答案
    
    Returns:
        is_correct: Hybrid 结果是否正确
    """
    # 计算有效接受长度
    L = compute_effective_length(accept_decisions)
    
    # 构建 Hybrid 序列
    # hybrid = draft[:L] + target[L:]
    hybrid_tokens = draft_tokens[:L] + target_tokens[L:]
    
    # 简化的正确性判断:
    # - 如果接受了太多 Mismatch，可能破坏正确性
    # - 实际使用时应该解码并验证答案
    
    mismatch_accepted = 0
    for i in range(min(L, len(draft_tokens))):
        if i < len(target_tokens) and draft_tokens[i] != target_tokens[i]:
            mismatch_accepted += 1
    
    # 启发式: 如果接受的 Mismatch 超过 50%，认为可能出错
    # 这是一个简化的判断，实际应该用真正的验证函数
    if L > 0 and mismatch_accepted / L > 0.5:
        return False
    
    return True


def compute_score(
    solution_str: str,
    ground_truth: str,
    draft_tokens: list = None,
    target_tokens: list = None,
    is_correct_baseline: bool = False,
    draft_len: int = 0,
    alpha: float = 1.0,
    penalty_break: float = -10.0,
    reward_correct: float = 100.0,
    reward_useless: float = 0.0,
    **kwargs
) -> float:
    """
    计算 SPD Scorer 的 Reward
    
    这是 verl 框架调用的主函数
    
    Args:
        solution_str: 模型生成的 "响应" (在 SPD 场景中，这是 Accept/Reject 决策序列)
        ground_truth: 正确答案
        draft_tokens: Draft token IDs
        target_tokens: Target token IDs
        is_correct_baseline: Target Model 是否答对
        draft_len: Draft 序列长度
        alpha: 加速奖励系数
        penalty_break: 破坏正确答案的惩罚
        reward_correct: 纠正错误的奖励
        reward_useless: 无用尝试的奖励
    
    Returns:
        reward: 计算得到的奖励值
    """
    try:
        # 解析模型的输出
        # 在 SPD 场景中，模型输出应该是 Accept/Reject 决策
        # 格式: "1 1 1 0 1 0 ..." 或 "[1, 1, 1, 0, 1, 0, ...]"
        
        if solution_str.startswith('['):
            # JSON 列表格式
            accept_decisions = json.loads(solution_str)
        else:
            # 空格分隔格式
            parts = solution_str.strip().split()
            accept_decisions = [int(p) > 0 for p in parts if p.isdigit()]
        
        # 如果解析失败，使用默认决策（全部接受）
        if not accept_decisions:
            accept_decisions = [True] * draft_len
        
        # 确保长度匹配
        if len(accept_decisions) < draft_len:
            accept_decisions.extend([False] * (draft_len - len(accept_decisions)))
        accept_decisions = accept_decisions[:draft_len]
        
    except Exception as e:
        logger.warning(f"解析 Accept/Reject 决策失败: {e}, 使用默认值")
        accept_decisions = [True] * draft_len
    
    # 计算有效接受长度
    L = compute_effective_length(accept_decisions)
    
    # 验证 Hybrid 正确性
    is_correct_hybrid = verify_hybrid_correctness(
        draft_tokens=draft_tokens or [],
        target_tokens=target_tokens or [],
        accept_decisions=accept_decisions,
        ground_truth=ground_truth
    )
    
    # 根据四场景计算奖励
    if is_correct_baseline and is_correct_hybrid:
        # 场景 A: 加速成功
        reward = alpha * L
        scenario = "A"
    elif is_correct_baseline and not is_correct_hybrid:
        # 场景 B: 破坏正确答案 (严厉惩罚)
        reward = penalty_break
        scenario = "B"
    elif not is_correct_baseline and not is_correct_hybrid:
        # 场景 C: 无用尝试
        reward = reward_useless
        scenario = "C"
    else:  # not is_correct_baseline and is_correct_hybrid
        # 场景 D: 纠正错误 (重奖)
        reward = reward_correct
        scenario = "D"
    
    # 返回结果
    # verl 期望返回 float 或包含 'score' 的 dict
    return {
        "score": reward,
        "effective_length": L,
        "scenario": scenario,
        "is_correct_hybrid": is_correct_hybrid,
        "accept_ratio": sum(accept_decisions) / len(accept_decisions) if accept_decisions else 0,
    }
'''
    
    # 确保 reward_score 目录存在
    reward_dir = os.path.join(os.getcwd(), "verl", "utils", "reward_score")
    os.makedirs(reward_dir, exist_ok=True)
    
    # 写入 reward function 文件
    reward_file = os.path.join(reward_dir, "spd_scorer_reward.py")
    with open(reward_file, 'w', encoding='utf-8') as f:
        f.write(reward_fn_code)
    
    logger.info(f"SPD Reward Function 已写入: {reward_file}")
    
    # 修改 __init__.py 以注册新的 reward function
    init_file = os.path.join(reward_dir, "__init__.py")
    
    # 检查是否已经注册
    if os.path.exists(init_file):
        with open(init_file, 'r', encoding='utf-8') as f:
            init_content = f.read()
        
        # 检查是否已经包含 spd_scorer
        if 'spd_scorer' not in init_content:
            # 找到合适的位置插入
            # 在 default_compute_score 函数中添加 spd_scorer 的处理
            
            insert_code = '''
    elif data_source == "spd_scorer":
        from . import spd_scorer_reward
        res = spd_scorer_reward.compute_score(
            solution_str, 
            ground_truth,
            draft_tokens=extra_info.get('draft_tokens') if extra_info else None,
            target_tokens=extra_info.get('target_tokens') if extra_info else None,
            is_correct_baseline=extra_info.get('is_correct_baseline', False) if extra_info else False,
            draft_len=extra_info.get('draft_len', 0) if extra_info else 0,
            alpha=extra_info.get('alpha', 1.0) if extra_info else 1.0,
            penalty_break=extra_info.get('penalty_break', -10.0) if extra_info else -10.0,
            reward_correct=extra_info.get('reward_correct', 100.0) if extra_info else 100.0,
            reward_useless=extra_info.get('reward_useless', 0.0) if extra_info else 0.0,
        )
'''
            
            # 在 "openai/gsm8k" 条件之前插入
            if 'elif data_source == "openai/gsm8k"' in init_content:
                init_content = init_content.replace(
                    'if data_source == "openai/gsm8k"',
                    f'if data_source == "spd_scorer":{insert_code[insert_code.find("from"):]}\n    elif data_source == "openai/gsm8k"'
                )
            else:
                # 如果找不到，尝试在函数开头插入
                logger.warning("无法自动注册 SPD reward function，请手动修改 verl/utils/reward_score/__init__.py")
            
            # 写回文件
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(init_content)
            
            logger.info("SPD Reward Function 已注册到 verl")
    
    return reward_file


# ==============================================================================
# 4. 训练主函数
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


def run_training(config: SPDTrainingConfig):
    """
    运行 SPD Scorer 训练
    
    完整流程:
        1. 准备训练数据
        2. 注册自定义 Reward Function
        3. 构建并执行训练命令
    
    Args:
        config: 训练配置
    """
    logger.info("=" * 60)
    logger.info("SPD Scorer GRPO 训练")
    logger.info("=" * 60)
    
    # Step 1: 准备数据
    logger.info("\n[Step 1] 准备训练数据...")
    train_file, val_file = prepare_spd_training_data(config)
    
    # Step 2: 注册 Reward Function
    logger.info("\n[Step 2] 注册自定义 Reward Function...")
    register_spd_reward_function()
    
    # Step 3: 构建训练命令
    logger.info("\n[Step 3] 构建训练命令...")
    cmd = build_training_command(config, train_file, val_file)
    
    logger.info("\n训练命令:")
    logger.info(" ".join(cmd[:5]) + " \\")
    for arg in cmd[5:]:
        logger.info(f"    {arg} \\")
    
    # Step 4: 执行训练
    logger.info("\n[Step 4] 启动训练...")
    logger.info("=" * 60)
    
    offload_status = "ON" if config.offload else "OFF"
    logger.info(f"🚀 开始 SPD Scorer GRPO 训练")
    logger.info(f"配置: {config.n_gpus} GPU | Batch={config.train_batch_size} | Rollout N={config.rollout_n} | Offload={offload_status}")
    logger.info("=" * 60)
    
    try:
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        env["NCCL_P2P_DISABLE"] = "1"
        
        subprocess.run(cmd, check=True, env=env)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n训练过程中出错: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断。")


# ==============================================================================
# 5. 独立的 SPD Scorer 训练循环 (不依赖 verl 的 main_ppo)
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
# 6. 命令行入口
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
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="生成的模拟样本数量")
    parser.add_argument("--source_data", type=str, default=None,
                        help="真实数据源路径 (可选)")
    
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
    if args.mode == "verl":
        run_training(config)
    else:
        train_spd_scorer_standalone(config)


if __name__ == "__main__":
    main()

