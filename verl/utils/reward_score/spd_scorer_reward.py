# -*- coding: utf-8 -*-
"""
SPD Scorer 自定义 Reward Function (Tensor Batch Optimized)
用于 verl 框架的 reward 计算

设计变更:
    - 移除 async/aiohttp，改为同步的 Tensor 批处理 + 批量 API 调用
    - 移除 parse_accept_decisions，直接使用 Tensor 输入
    - 强制使用 Batch 模式，大幅提升 GPU 利用率和吞吐量
    - 使用离线 vLLM (LLM.generate) 替代 HTTP API，支持单例缓存

环境变量配置:
    # ========== 必需 ==========
    SPD_TARGET_MODEL_PATH       : Target Model 路径 (用于 vLLM 和 Tokenizer)
    
    # ========== Reward 计算参数 ==========
    SPD_REWARD_ALPHA            : 线性奖励系数 (默认: 1.0)
    SPD_REWARD_PENALTY_BREAK    : 中断惩罚 (默认: -10.0)
    SPD_REWARD_CORRECT          : 意外答对奖励 (默认: 100.0)
    SPD_REWARD_USELESS          : 无用惩罚 (默认: 0.0)
    
    # ========== vLLM 引擎配置 ==========
    SPD_VLLM_TENSOR_PARALLEL_SIZE      : 张量并行大小 (默认: 1)
    SPD_VLLM_GPU_MEMORY_UTILIZATION    : GPU 显存利用率 (默认: 0.9)
    SPD_VLLM_MAX_MODEL_LEN             : 最大模型长度 (默认: 不限制)
    
    # ========== vLLM 生成参数 ==========
    SPD_VLLM_TEMPERATURE        : 采样温度 (默认: 0.0)
    SPD_VLLM_MAX_TOKENS         : 最大生成 token 数 (默认: 1024)

作者: AI Assistant
日期: 2025-11-26
"""

import logging
import torch
import numpy as np
import os
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# ============================================================================
# 全局缓存 (单例模式)
# ============================================================================
_TOKENIZER_CACHE = {}
_VLLM_ENGINE_CACHE = {}  # 缓存 vLLM LLM 实例

def get_tokenizer(model_path: str):
    """获取或加载 Tokenizer"""
    if model_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_path]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer


def get_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    **kwargs
):
    """
    获取或初始化 vLLM LLM 实例 (单例模式)
    
    Args:
        model_path: 模型路径或 HuggingFace 模型名
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU 显存利用率
        max_model_len: 最大模型长度
        trust_remote_code: 是否信任远程代码
        dtype: 数据类型
        **kwargs: 其他 vLLM 参数
    
    Returns:
        vLLM LLM 实例
    """
    # 使用 model_path 作为 cache key
    cache_key = model_path
    
    if cache_key in _VLLM_ENGINE_CACHE:
        logger.info(f"[vLLM] Reusing cached engine for: {model_path}")
        return _VLLM_ENGINE_CACHE[cache_key]
    
    logger.info(f"[vLLM] Initializing new engine for: {model_path}")
    
    from vllm import LLM
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        **kwargs
    )
    
    _VLLM_ENGINE_CACHE[cache_key] = llm
    logger.info(f"[vLLM] Engine initialized and cached for: {model_path}")
    
    return llm


def batch_vllm_generate(
    prompt_token_ids: List[List[int]],
    model_path: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    # vLLM 引擎配置参数
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    **engine_kwargs
) -> List[str]:
    """
    使用离线 vLLM 批量生成 (直接传入 Token IDs)
    
    Args:
        prompt_token_ids: 批量 prompt token ids, List[List[int]]
        model_path: 模型路径
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        top_p: nucleus sampling 参数
        tensor_parallel_size: 张量并行大小 (首次初始化时使用)
        gpu_memory_utilization: GPU 显存利用率 (首次初始化时使用)
        max_model_len: 最大模型长度 (首次初始化时使用)
        **engine_kwargs: 其他 vLLM 引擎参数
    
    Returns:
        生成的文本列表
    """
    if not prompt_token_ids:
        return []
    
    from vllm import SamplingParams
    
    # 获取或初始化 vLLM 引擎 (单例)
    llm = get_vllm_engine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        **engine_kwargs
    )
    
    # 构造采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # 批量生成 - 直接传入 token ids
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    
    # 提取生成的文本
    results = []
    for output in outputs:
        # output.outputs[0] 是第一个生成结果 (因为 n=1)
        generated_text = output.outputs[0].text
        results.append(generated_text)
    
    return results
        
    


def compute_score(
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    prompt_ids: Optional[torch.Tensor] = None,   # [Batch, Seq_Len]
    response_ids: Optional[torch.Tensor] = None, # [Batch, Response_Len] (即 Accept Decisions)
    attention_mask: Optional[torch.Tensor] = None,
    # Reward 计算参数
    alpha: Optional[float] = None,
    penalty_break: Optional[float] = None,
    reward_correct: Optional[float] = None,
    reward_useless: Optional[float] = None,
    # vLLM 模型配置
    model_path: Optional[str] = None,
    # 其他参数 (包括 vLLM 引擎配置: tensor_parallel_size, gpu_memory_utilization, max_model_len 等)
    **kwargs
) -> List[Union[float, Dict[str, Any]]]:
    """
    批量计算 Reward (Tensor Batch Optimized)
    
    逻辑:
    1. 使用 response_ids (Tensor) 直接计算有效长度 L (Tensor Op)
    2. 使用 prompt_ids (Tensor) 提取 Context 和 Draft Tokens
    3. 构造 Hybrid Prefix (Context + Draft[:L]) -> Batch List[int]
    4. 批量调用 vLLM 生成
    5. 验证正确性 -> 计算 Reward (使用统一公式)
    
    完全向量化实现，移除 Python 循环。
    """
    
    # 从环境变量读取默认配置 (如果未通过参数传入)
    if alpha is None:
        alpha = float(os.getenv("SPD_REWARD_ALPHA", "1.0"))
    if penalty_break is None:
        penalty_break = float(os.getenv("SPD_REWARD_PENALTY_BREAK", "-10.0"))
    if reward_correct is None:
        reward_correct = float(os.getenv("SPD_REWARD_CORRECT", "100.0"))
    if reward_useless is None:
        reward_useless = float(os.getenv("SPD_REWARD_USELESS", "0.0"))
    
    # model_path 也可以从环境变量读取
    if model_path is None:
        model_path = os.getenv("SPD_TARGET_MODEL_PATH")

    # -------------------------------------------------------------------------
    # 0. 数据校验 (Strict Mode)
    # -------------------------------------------------------------------------
    if response_ids is None or prompt_ids is None:
        raise ValueError("compute_score_batch: 'response_ids' and 'prompt_ids' tensors are REQUIRED.")

    batch_size = prompt_ids.shape[0]
    
    # -------------------------------------------------------------------------
    # 从环境变量读取 vLLM 配置
    # -------------------------------------------------------------------------
    m_path = model_path or os.getenv("SPD_TARGET_MODEL_PATH")
    if not m_path:
        raise ValueError("Missing 'model_path'. Set SPD_TARGET_MODEL_PATH env var or pass model_path parameter.")

    tokenizer = get_tokenizer(m_path)
    
    # vLLM 引擎配置 (全部从环境变量读取)
    _tp_size = os.getenv("SPD_VLLM_TENSOR_PARALLEL_SIZE", "1")
    _gpu_util = os.getenv("SPD_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
    _max_len = os.getenv("SPD_VLLM_MAX_MODEL_LEN", "")
    
    vllm_config = {
        "tensor_parallel_size": int(_tp_size),
        "gpu_memory_utilization": float(_gpu_util),
        "max_model_len": int(_max_len) if _max_len else None,
    }

    # -------------------------------------------------------------------------
    # 1. 计算有效长度 L (Tensor Parallel)
    # -------------------------------------------------------------------------
    accept_decisions = response_ids.float()
    valid_mask = torch.cumprod(accept_decisions, dim=1)
    L_tensor = valid_mask.sum(dim=1).long() # [Batch]
    L_list = L_tensor.tolist()

    # -------------------------------------------------------------------------
    # 2. 构造 vLLM Prompts & Baseline Correctness
    # -------------------------------------------------------------------------
    # 直接从 extra_infos 提取数据，不再 Parse prompt_ids
    
    context_ids_list = []
    draft_tokens_list = []
    is_correct_baseline_list = []
    target_tokens_list = []
    bonus_tokens_list = []
    
    for info in extra_infos:
        # Context IDs (List[int])
        c_ids = info.get("context_ids")
        if c_ids is None:
            raise ValueError("Missing 'context_ids' in extra_info")
        context_ids_list.append(c_ids)
        
        # Draft Tokens (List[int])
        d_tokens = info.get("draft_tokens")
        if d_tokens is None:
            raise ValueError("Missing 'draft_tokens' in extra_info")
        draft_tokens_list.append(d_tokens)

        target_tokens = info.get("target_tokens")
        if target_tokens is None:
            raise ValueError("Missing 'target_tokens' in extra_info")
        target_tokens_list.append(target_tokens)

        bonus_tokens = info.get("bonus_tokens")
        if bonus_tokens is None:
            raise ValueError("Missing 'bonus_tokens' in extra_info")
        bonus_tokens_list.append(bonus_tokens)
        
        # Baseline Correctness
        is_correct = info.get("is_correct_baseline")
        if is_correct is None:
            raise ValueError("Missing 'is_correct_baseline' in extra_info")
        is_correct_baseline_list.append(is_correct)
    
    # 转为 Tensor 以便后续计算
    s_t_tensor = torch.tensor([1.0 if x else 0.0 for x in is_correct_baseline_list], device=response_ids.device)

    # 构造 Prompts
    prompts_for_vllm = []
    
    for i in range(batch_size):
        l_val = L_list[i]
        
        # 如果 L=0，Hybrid == Baseline，无需调用 API
        if l_val == 0:
            prompts_for_vllm.append(None)
        else:
            if l_val < len(draft_tokens_list[i]):
                # Context + Draft[:L]
                # 这里是 List 拼接
                hybrid_ids = context_ids_list[i] + draft_tokens_list[i][:l_val] + target_tokens_list[i][l_val:l_val+1]
                prompts_for_vllm.append(hybrid_ids)
            else:
                # Context + Draft + Bonus
                # 这里是 List 拼接
                hybrid_ids = context_ids_list[i] + draft_tokens_list[i] + bonus_tokens_list[i]
                prompts_for_vllm.append(hybrid_ids)
            

    # -------------------------------------------------------------------------
    # 3. 批量生成 (使用离线 vLLM)
    # -------------------------------------------------------------------------
    valid_prompts = [p for p in prompts_for_vllm if p is not None]
    completions = []
    
    if valid_prompts:
        # 生成参数也从环境变量读取
        _temperature = float(os.getenv("SPD_VLLM_TEMPERATURE", "0.0"))
        _max_tokens = int(os.getenv("SPD_VLLM_MAX_TOKENS", "1024"))
        
        completions = batch_vllm_generate(
            prompt_token_ids=valid_prompts,
            model_path=m_path,
            temperature=_temperature,
            max_tokens=_max_tokens,
            **vllm_config
        )
    
    # -------------------------------------------------------------------------
    # 4. 验证正确性 & 计算 Reward (Vectorized)
    # -------------------------------------------------------------------------
    # 我们需要计算 s_h (Score Hybrid)。
    # 对于 L=0 的样本，s_h = s_t。
    # 对于 L>0 的样本，检查 ground_truth in (Accepted + Completion)。
    
    s_h_list = []
    comp_idx = 0
    
    for i in range(batch_size):
        if prompts_for_vllm[i] is None:
            # L=0 case: 如果一个都没接收, 那么就认为没做对
            s_h_list.append(0)
        else:
            # 获取 prompt (context + draft[:l_val] + ...) 和 completion
            prompt_text = tokenizer.decode(valid_prompts[comp_idx], skip_special_tokens=True)
            completion_text = completions[comp_idx]
            comp_idx += 1
            
            # 完整答案 = prompt + completion
            final_answer = prompt_text + completion_text
            gt = ground_truths[i]
            
            if "\\boxed" in gt:
                is_correct = gt in final_answer
            else:
                is_correct = gt.strip() in final_answer.strip()
            
            s_h_list.append(1.0 if is_correct else 0.0)

    s_h_tensor = torch.tensor(s_h_list, device=response_ids.device)

    # -------------------------------------------------------------------------
    # 5. 应用公式计算 Reward (Pure Tensor Op)
    # -------------------------------------------------------------------------
    # R = S_t * S_h * (alpha * L) + 
    #     S_t * (1 - S_h) * penalty_break + 
    #     (1 - S_t) * (1 - S_h) * reward_useless + 
    #     (1 - S_t) * S_h * reward_correct
    
    # Cast L_tensor to float for multiplication
    L_float = L_tensor.float()
    
    # 变量含义注释:
    # s_t_tensor: [B], target 是否做对 (答案和 target 完全匹配, 1/0)
    # s_h_tensor: [B], hybrid 形式是否做对 (包含 draft 被采纳+补全后, 是否答对, 1/0)
    # alpha: 线性奖励系数 (float, 奖励单元长度)
    # L_float: [B], draft 前缀有效采纳长度 (float)
    # penalty_break: 中途 break 时的奖励 (float, draft 不完全采纳但未答对)
    # reward_useless: 完全没答对时的奖励 (float, 没采纳且没答对)
    # reward_correct: 没采纳 draft 但偶然完全答对时的奖励 (float)
    #
    # 各项分别代表以下 case:
    # term1: target 做对且 hybrid 做对, 给与线性奖励 (鼓励采纳 draft 并答对)
    # term2: target 做对但 hybrid 没做对, 给 penalty（采纳但没答对）
    # term3: target 没做对且 hybrid 也没做对, 给“useless”奖励
    # term4: target 没做对但 hybrid 做对, 给“奇迹”奖励
    term1 = s_t_tensor * s_h_tensor * (alpha * L_float)
    term2 = s_t_tensor * (1.0 - s_h_tensor) * penalty_break
    term3 = (1.0 - s_t_tensor) * (1.0 - s_h_tensor) * reward_useless
    term4 = (1.0 - s_t_tensor) * s_h_tensor * reward_correct
    
    rewards_tensor = term1 + term2 + term3 + term4
    
    # Return list of results (or tensor if caller supports it, but BatchRewardManager expects list of dicts/floats)
    # BatchRewardManager expects a list of results.
    # We can return simple floats or dicts. To match interface, let's return list of dicts.
    # But now we don't loop for calculation! We loop only for formatting return.
    
    # 快速构造返回结果
    rewards_list = rewards_tensor.tolist()
    
    return [
        {
            "score": r,
            "effective_length": l,
            # Skip scenario string generation for speed if not needed
        }
        for r, l in zip(rewards_list, L_list)
    ]

