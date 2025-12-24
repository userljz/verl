# -*- coding: utf-8 -*-
"""
SPD Scorer 自定义 Reward Function (Lightweight Version)
用于 verl 框架的 reward 计算

设计说明:
    - Rollout 阶段计算 effective_len 和 is_correct_hybrid，作为 Tensor 放入 batch 中
    - is_correct_baseline 从 extra_info 中读取 (数据集原始字段)

数据流:
    - extra_info (来自数据集): is_correct_baseline 等原始信息

作者: AI Assistant
日期: 2025-12-06
"""

import os
import sys
from typing import Dict, Any, Optional

import torch.distributed as dist
from loguru import logger


# ==============================================================================
# Loguru Rank0-Only 初始化（与 train_spd_scorer.py 保持一致）
# 注意: 此文件在 Ray Trainer Driver 进程中执行，与 spd_rollout.py 不在同一进程
# ==============================================================================
_LOGURU_RANK0_INIT = False

def _setup_loguru_rank0():
    """
    配置 loguru 只让 rank0 输出 DEBUG/INFO，WARNING/ERROR 全 rank 输出。
    输出到环境变量 SPD_LOG_FILE 指定的文件（与 train_spd_scorer.py 保持一致）。
    该函数只会执行一次（幂等）。
    """
    global _LOGURU_RANK0_INIT
    if _LOGURU_RANK0_INIT:
        return
    
    log_file = os.getenv("SPD_LOG_FILE")
    if not log_file:
        # 如果没有设置 SPD_LOG_FILE，跳过配置（保持默认行为）
        return
    
    def _filter(record):
        # WARNING/ERROR 全 rank 输出；DEBUG/INFO 只 rank0 输出
        if record["level"].no >= 30:
            return True
        return (not dist.is_initialized()) or dist.get_rank() == 0

    level = os.getenv("LOGURU_LEVEL", "INFO")
    
    logger.remove()
    # 输出到文件，与 train_spd_scorer.py 保持一致
    logger.add(log_file, level=level, filter=_filter)
    
    _LOGURU_RANK0_INIT = True

# 模块加载时自动初始化
_setup_loguru_rank0()


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """
    计算 SPD Scorer 的 Reward
    
    Args:
        solution_str: 模型的输出字符串 (此处不用)
        ground_truth: 答案真值 (此处不用)
        extra_info: 数据集原始信息，包含 'is_correct_baseline' 等
        **kwargs: 其他参数 (未使用)
        
    Returns:
        float: Reward 分数
        
    Reward 公式:
        R = S_t * S_h * (alpha * L) +                              # 场景A: 原本正确，Hybrid 也正确 -> 奖励 L
            S_t * (1 - S_h) * penalty_break +                      # 场景B: 原本正确，Hybrid 出错 -> 惩罚
            (1 - S_t) * (1 - S_h) * reward_useless +               # 场景C: 原本错误，Hybrid 也错误 -> 无影响
            (1 - S_t) * S_h * (reward_correct_base + alpha * L)    # 场景D: 原本错误，Hybrid 变正确 -> 基础奖励 + 长度奖励
    """
    
    
    # =================================================================
    # 1. 从 extra_info (数据集原始字段 + Rollout计算结果) 读取数据
    # =================================================================
    # ==================== DEBUG: Reward 计算开始 ====================
    logger.debug("=" * 50)
    logger.debug("[SPD Reward] 开始计算 Reward")
    logger.debug(f"[SPD Reward] extra_info keys: {list(extra_info.keys()) if extra_info else 'None'}")

    # 从 extra_info 中读取 effective_len (L)
    # Rollout 阶段已将 effective_len 注入到 extra_info 中
    L = extra_info.get('effective_len', 0.0)
    L = float(L)
    
    # ==================== DEBUG: 读取 L ====================
    logger.debug(f"[SPD Reward] 有效长度 L = {L}")

    # =================================================================
    # [Fix] L=0 时直接返回 0，不参与学习
    # 原因: 如果没有接受任何 draft token，无法从中学到任何东西
    # =================================================================
    if L == 0:
        logger.debug(f"[SPD Reward] L=0, 跳过计算 (reward=0.0)")
        return 0.0

    # 从 extra_info 中读取 is_correct_hybrid (S_h)
    s_h_raw = extra_info.get('is_correct_hybrid', False)
    s_h = 1.0 if s_h_raw else 0.0

    # 从 extra_info 中读取 baseline 验证结果 (S_t)
    s_t_raw = extra_info.get('is_correct_baseline', False)
    s_t = 1.0 if s_t_raw else 0.0
    
    # ==================== DEBUG: 正确性信息 ====================
    logger.debug(f"[SPD Reward] is_correct_baseline (S_t) = {s_t_raw} -> {s_t}")
    logger.debug(f"[SPD Reward] is_correct_hybrid (S_h) = {s_h_raw} -> {s_h}")


    # =================================================================
    # 3. 读取 Reward 系数 (环境变量优先)
    # =================================================================
    alpha = float(os.getenv("SPD_REWARD_ALPHA", "1.0"))
    penalty_break = float(os.getenv("SPD_REWARD_PENALTY_BREAK", "-10.0"))
    reward_correct_base = float(os.getenv("SPD_REWARD_CORRECT_BASE", "5.0"))  # 场景D的基础奖励
    reward_useless = float(os.getenv("SPD_REWARD_USELESS", "0.0"))
    
    # ==================== DEBUG: Reward 系数 ====================
    logger.debug(f"[SPD Reward] Reward 系数: alpha={alpha}, penalty_break={penalty_break}, reward_correct_base={reward_correct_base}, reward_useless={reward_useless}")
    
    # =================================================================
    # 4. 应用 Reward 公式
    # =================================================================
    # R = S_t * S_h * (alpha * L) +                              # 场景A: 加速成功
    #     S_t * (1 - S_h) * penalty_break +                      # 场景B: 破坏正确
    #     (1 - S_t) * (1 - S_h) * reward_useless +               # 场景C: 无用尝试
    #     (1 - S_t) * S_h * (reward_correct_base + alpha * L)    # 场景D: 纠正错误 (基础奖励 + 长度奖励)
    
    term1 = s_t * s_h * (alpha * L)
    term2 = s_t * (1.0 - s_h) * penalty_break
    term3 = (1.0 - s_t) * (1.0 - s_h) * reward_useless
    term4 = (1.0 - s_t) * s_h * (reward_correct_base + alpha * L)  # 新公式: 基础奖励 + 长度奖励
    
    reward = term1 + term2 + term3 + term4
    
    # ==================== DEBUG: Reward 公式分解 ====================
    # 判断当前样本属于哪个场景
    if s_t == 1.0 and s_h == 1.0:
        scenario = "A (加速成功: 原本正确, Hybrid也正确)"
        active_term = f"term1 = {s_t} * {s_h} * ({alpha} * {L}) = {term1:.2f}"
    elif s_t == 1.0 and s_h == 0.0:
        scenario = "B (破坏正确: 原本正确, Hybrid错误)"
        active_term = f"term2 = {s_t} * (1-{s_h}) * {penalty_break} = {term2:.2f}"
    elif s_t == 0.0 and s_h == 0.0:
        scenario = "C (无用尝试: 原本错误, Hybrid也错误)"
        active_term = f"term3 = (1-{s_t}) * (1-{s_h}) * {reward_useless} = {term3:.2f}"
    else:  # s_t == 0.0 and s_h == 1.0
        scenario = "D (纠正错误: 原本错误, Hybrid正确)"
        active_term = f"term4 = (1-{s_t}) * {s_h} * ({reward_correct_base} + {alpha}*{L}) = {term4:.2f}"
    
    logger.debug(f"[SPD Reward] 场景: {scenario}")
    logger.debug(f"[SPD Reward] 激活项: {active_term}")
    logger.debug(f"[SPD Reward] 各项值: term1={term1:.2f}, term2={term2:.2f}, term3={term3:.2f}, term4={term4:.2f}")
    logger.debug(f"[SPD Reward] 最终 Reward: {reward:.2f}")
    logger.debug("=" * 50)
    
    return float(reward)
